"""Label-free recovery of the target operating threshold — CPU pilot (2026-09-12).

Question. operating_points.json / threshold_decomposition.json show that under
dataset shift most of a YOLO detector's "miscalibration" is the SOURCE-selected
confidence threshold t_src being wrong for the target, and that re-selecting
the threshold on the target WITH labels repairs most of it. A practitioner has
no target labels. Can the target-appropriate threshold be recovered from the
UNLABELLED target score distribution alone?

Protocol (per cross-dataset cell of operating_points.json):
  source calval  : detections + labels  -> t_src, TP/FP score components
  target calval  : detections, SCORES ONLY (labels touched only by ORACLE and
                   by the pi_true diagnostic)          = unlabelled adaptation set
  target test    : detections + labels                 = evaluation set
Every target listed has its own calval/test split, so no random half split was
needed (FIgLib has no calval split and is not a listed target; excluded).

Estimators (all label-free on the target):
  M0   t_src                                  source F1-optimal (what ships)
  D25  0.25                                   ultralytics default, for reference
  M1a  quantile matching, pooled scores       F_tgt^-1(F_src(t_src))
  M1b  quantile matching, per-image max       = alarm-rate matching
  M1c  kept-detections-per-image matching     N_kept/N_images preserved
  M2   two-component mixture (Saerens-style prior shift): TP and FP score
       densities fitted on the source calval (Beta by moments, Beta by MLE,
       20-bin logit-spaced histogram); EM over the mixing weight pi only on
       the target unlabelled pooled scores; threshold at posterior P(TP|s)=0.5
       ("post05") and at the mixture-implied F1 optimum ("mixf1"; the
       undetected-GT count is unknown label-free and is transferred from the
       source as n_gt/n_TP-at-floor).
  M3   M2 (Beta-MLE) with a shared logit-space location shift delta of both
       components, chosen by maximum marginal likelihood on the target.
  ORC  oracle: F1-optimal threshold on the target calval WITH labels.
  TOPT test-optimal: F1-optimal on the test split itself (true ceiling).
Reported per cell and method: threshold, box F1 / P / R at IoU 0.5 on the
target test, image-level sensitivity and false-alarm rate, LaECE_0 (identity
map, detections kept at the threshold; toolbox convention of
threshold_decomposition.py's thr=*|map=none cells), regret = F1(ORC) - F1(m).
Shift diagnostic: 1-D Wasserstein distance between source-calval and target
unlabelled score distributions (pooled, per-image max) and its Spearman
correlation across cells with the F1 loss at t_src and with |t_src - t_orc|.

Matching / PR / image-level functions are IMPORTED from operating_points.py
unchanged (same conventions: greedy score-ordered per-class matching at IoU
0.5, single threshold across the evaluated classes cls_t = target-annotated
and model-predicted classes). Writes analysis/label_free_threshold_pilot.json
(+ .md). CPU only.
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import expit, logit, logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operating_points as op  # noqa: E402  (read-only reuse)

GT, DETS = HERE / "coco_gt", HERE / "detections"
TARGET_CALVAL = {
    "d_fire_test": "d_fire_calval", "d_fire_test_smokeonly": "d_fire_calval",
    "d_fire_test_dedup": "d_fire_calval_dedup", "d_fire_test_smokeonly_dedup": "d_fire_calval_dedup",
    "pyro_sdis_caltest": "pyro_sdis_calval",
    "thesis_test": "thesis_calval", "thesis_test_smokeonly": "thesis_calval",
}
EPS = 5e-4            # score clipping for Beta fits / logit grids (export floor is conf = 0.001)
N_HIST = 20           # histogram component bins (logit-spaced)
GRID = expit(np.linspace(logit(EPS), logit(1 - EPS), 2000))   # threshold grid for mixture rules
DELTA_GRID = np.linspace(-3.0, 3.0, 61)
# LaECE_0 (toolbox) only for the main methods, and only when the kept set is not huge (RT-DETR exports 300 dets/image)
LAECE_METHODS = {"M0_t_src", "D25_default", "M1a_quantile_pooled", "M1b_quantile_imgmax", "M1c_kept_rate",
                 "M2_beta_mle_post05", "M2_beta_mle_mixf1", "M3_beta_mle_shift_post05", "M3_hist20_shift_post05",
                 "ORC_target_calval"}
LAECE_MAX_DETS = 80_000


def dataset_of(tag):
    for k in ("d_fire", "pyro_sdis", "thesis", "figlib"):
        if tag.startswith(k):
            return k
    return tag


def _q(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ----------------------------------------------------------------------------- score components
class BetaComp:
    def __init__(self, x, how):
        x = np.clip(x, EPS, 1 - EPS)
        if how == "mom":
            m, v = x.mean(), x.var()
            c = m * (1 - m) / max(v, 1e-12) - 1
            self.a, self.b = max(m * c, 1e-3), max((1 - m) * c, 1e-3)
        else:
            self.a, self.b, _, _ = stats.beta.fit(x, floc=0, fscale=1)
        self.n = int(len(x))

    def logpdf(self, s):
        return stats.beta.logpdf(np.clip(s, EPS, 1 - EPS), self.a, self.b)

    def cdf(self, s):
        return stats.beta.cdf(np.clip(s, EPS, 1 - EPS), self.a, self.b)

    def params(self):
        return {"family": "beta", "a": float(self.a), "b": float(self.b), "n_fit": self.n}


class HistComp:
    edges = expit(np.linspace(logit(EPS), logit(1 - EPS), N_HIST + 1))

    def __init__(self, x):
        x = np.clip(x, EPS, 1 - EPS)
        cnt = np.histogram(x, bins=self.edges)[0].astype(float) + 0.5   # Laplace-smoothed
        self.p = cnt / cnt.sum()
        self.width = np.diff(self.edges)
        self.n = int(len(x))

    def _bin(self, s):
        return np.clip(np.searchsorted(self.edges, np.clip(s, EPS, 1 - EPS), side="right") - 1, 0, N_HIST - 1)

    def logpdf(self, s):
        j = self._bin(s)
        return np.log(self.p[j]) - np.log(self.width[j])

    def cdf(self, s):
        j = self._bin(s)
        cum = np.concatenate([[0.0], np.cumsum(self.p)])
        frac = (np.clip(s, EPS, 1 - EPS) - self.edges[j]) / self.width[j]
        return cum[j] + self.p[j] * np.clip(frac, 0, 1)

    def params(self):
        return {"family": "hist20_logit", "edges": [float(e) for e in self.edges], "p": [float(v) for v in self.p], "n_fit": self.n}


def em_pi(lf_tp, lf_fp, pi0, w=None, iters=1000, tol=1e-8):
    """EM over the mixing weight only; components fixed. Returns (pi, loglik, n_iter)."""
    w = np.ones_like(lf_tp) if w is None else w
    pi = float(np.clip(pi0, 1e-4, 1 - 1e-4))
    for it in range(1, iters + 1):
        a, b = np.log(pi) + lf_tp, np.log1p(-pi) + lf_fp
        r = np.exp(a - np.logaddexp(a, b))
        new = float(np.clip(np.sum(w * r) / np.sum(w), 1e-6, 1 - 1e-6))
        done = abs(new - pi) < tol
        pi = new
        if done:
            break
    ll = float(np.sum(w * np.logaddexp(np.log(pi) + lf_tp, np.log1p(-pi) + lf_fp)))
    return pi, ll, it


def mixture_thresholds(comp_tp, comp_fp, pi, n_dets, r_src, s_max_src, delta=0.0):
    """Posterior-0.5 and mixture-F1-optimal thresholds on GRID (target score space).
    delta: logit shift mapping target scores back to source space (M3). The
    posterior scan is confined to the source score support (<= s_max_src in
    source space): above it the components are unobserved (histogram bins hold
    only the Laplace pseudo-count) and inherit the last supported posterior."""
    g_src = expit(logit(GRID) - delta)
    lt, lf = comp_tp.logpdf(g_src), comp_fp.logpdf(g_src)
    post = expit((np.log(pi) + lt) - (np.log1p(-pi) + lf))
    sup = g_src <= s_max_src
    if sup.any():
        post[~sup] = post[sup][-1]
    below = np.where(post < 0.5)[0]
    if len(below) == 0:
        t05 = float(GRID[0])
    elif below[-1] == len(GRID) - 1:
        t05 = 1.0
    else:
        t05 = float(GRID[below[-1] + 1])
    tp_k = n_dets * pi * (1 - comp_tp.cdf(g_src))
    fp_k = n_dets * (1 - pi) * (1 - comp_fp.cdf(g_src))
    n_gt_hat = r_src * n_dets * pi
    f1 = 2 * tp_k / np.maximum(tp_k + fp_k + n_gt_hat, 1e-12)
    j = int(np.argmax(f1))
    return t05, float(GRID[j]), float(f1[j]), float(n_gt_hat)


def laece0(test_gt, dets):
    """LaECE_0 of the identity map on the detections kept (mirrors calib_metrics.evaluate, first evaluator)."""
    from detection_calibration.coco_calibration import CalibrationCOCO
    from pycocotools.coco import COCO
    if not dets:
        return None
    ev = CalibrationCOCO(test_gt, test_gt, "bbox", 25, 0.0, False, False, 100)
    ev.cocoDt = _q(COCO(test_gt).loadRes, dets)
    _q(ev.evaluate); _q(ev.prepare_input); _q(ev.compute_single_errors)
    return float(ev.accumulate_errors())


# ----------------------------------------------------------------------------- cells
def select_cells(opj):
    cells, skipped = [], []
    names = set(opj["cells"])
    for name, e in opj["cells"].items():
        if name.endswith("__repair"):
            continue
        run, src, tgt = e["run"], e["source_calval"], e["target_test"]
        if dataset_of(src) == dataset_of(tgt):
            continue                                   # in-domain
        if tgt not in TARGET_CALVAL:
            skipped.append({"cell": name, "why": f"target {tgt} has no calval split / not a listed target"}); continue
        # pyrosdis-trained (smoke-only) models: <target> and <target>_smokeonly are the same evaluation
        if dataset_of(src) == "pyro_sdis" and not tgt.endswith("smokeonly") and f"{run}__to__{tgt}_smokeonly" in names:
            skipped.append({"cell": name, "why": f"identical to {run}__to__{tgt}_smokeonly (smoke-only model)"}); continue
        cal = TARGET_CALVAL[tgt]
        need = {"source_calval": DETS / f"{run}__{src}.bbox.json", "target_calval": DETS / f"{run}__{cal}.bbox.json",
                "target_test": DETS / f"{run}__{tgt}.bbox.json"}
        missing = [str(p) for p in need.values() if not p.exists()]
        if missing:
            skipped.append({"cell": name, "why": "missing detections", "files": missing}); continue
        cells.append((name, run, src, cal, tgt, {k: str(v) for k, v in need.items()}))
    return cells, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-laece", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on cell name (debug)")
    args = ap.parse_args()
    t_start = time.time()
    opj = json.loads((HERE / "operating_points.json").read_text(encoding="utf-8"))
    cells, skipped = select_cells(opj)
    if args.only:
        cells = [c for c in cells if args.only in c[0]]
    print(f"{len(cells)} cells, {len(skipped)} skipped")

    gt_cache, det_cache, match_cache = {}, {}, {}

    def gt(tag):
        if tag not in gt_cache:
            gt_cache[tag] = op.load_gt(tag)
        return gt_cache[tag]

    def dets(run, tag):
        k = (run, tag)
        if k not in det_cache:
            det_cache[k] = op.load_dets(DETS / f"{run}__{tag}.bbox.json")
        return det_cache[k]

    def matched(run, tag, classes):
        k = (run, tag, tuple(sorted(classes)))
        if k not in match_cache:
            ids, boxes, _ = gt(tag)
            match_cache[k] = op.match(ids, boxes, dets(run, tag), classes)
        return match_cache[k]

    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "iou": op.IOU, "eps": EPS, "n_hist_bins": N_HIST,
           "protocol": __doc__, "skipped": skipped, "cells": {}}
    rows = []
    for name, run, src, cal, tgt, files in cells:
        t0 = time.time()
        s_ids, s_gt, s_cls = gt(src)
        _, _, t_cls = gt(tgt)
        s_d, x_d = dets(run, src), dets(run, tgt)
        model_cls = {c for v in s_d.values() for c, _, _ in v} | {c for v in x_d.values() for c, _, _ in v}
        cls_s, cls_t = s_cls & model_cls, t_cls & model_cls
        # --- source calval (labelled) -------------------------------------------------
        sc, stp, sng, s_img = matched(run, src, cls_s)
        S = op.pr_curve(sc, stp, sng)
        t_src = float(S[0][int(np.argmax(S[3]))])
        t_src_ref = opj["cells"][name]["source"]["t_f1opt"]
        # --- target calval: scores only (labels used ONLY for ORC / pi_true) ------------
        uc, utp, ung, u_img = matched(run, cal, cls_t)
        u_max = np.array([m for m, _ in u_img])
        n_u_img = len(u_img)
        U = op.pr_curve(uc, utp, ung)
        t_orc = float(U[0][int(np.argmax(U[3]))])
        # --- target test (labelled, evaluation) ---------------------------------------
        xc, xtp, xng, x_img = matched(run, tgt, cls_t)
        X = op.pr_curve(xc, xtp, xng)
        t_topt = float(X[0][int(np.argmax(X[3]))])
        # --- source score statistics ---------------------------------------------------
        s_max = np.array([m for m, _ in s_img])
        q_pool = float(np.mean(sc < t_src))
        q_max = float(np.mean(s_max < t_src))
        kept_rate = float(np.sum(sc >= t_src) / len(s_img))
        pi_src = float(stp.mean())
        r_src = float(sng / max(stp.sum(), 1))

        thr = {"M0_t_src": t_src, "D25_default": 0.25}
        thr["M1a_quantile_pooled"] = float(np.quantile(uc, q_pool)) if len(uc) else 1.0
        thr["M1b_quantile_imgmax"] = float(np.quantile(u_max, q_max))
        k = int(round(kept_rate * n_u_img))
        us = np.sort(uc)[::-1]
        thr["M1c_kept_rate"] = float(us[min(k, len(us)) - 1]) if k >= 1 else 1.0

        # --- M2: mixture, EM over pi ------------------------------------------------------
        comps = {"beta_mom": (BetaComp(sc[stp], "mom"), BetaComp(sc[~stp], "mom")),
                 "beta_mle": (BetaComp(sc[stp], "mle"), BetaComp(sc[~stp], "mle")),
                 "hist20": (HistComp(sc[stp]), HistComp(sc[~stp]))}
        mix = {}
        for cname, (ctp, cfp) in comps.items():
            lt, lf = ctp.logpdf(uc), cfp.logpdf(uc)
            pi, ll, nit = em_pi(lt, lf, pi_src)
            t05, tf1, f1hat, ngt_hat = mixture_thresholds(ctp, cfp, pi, len(uc), r_src, float(sc.max()))
            thr[f"M2_{cname}_post05"], thr[f"M2_{cname}_mixf1"] = t05, tf1
            mix[cname] = {"pi_hat": pi, "loglik": ll, "em_iters": nit, "t_post05": t05, "t_mixf1": tf1,
                          "mixture_f1_hat": f1hat, "n_gt_hat": ngt_hat,
                          "tp_component": ctp.params(), "fp_component": cfp.params()}
        # --- M3: shared logit-space location shift delta of both components, binned target scores ----
        b_edges = expit(np.linspace(logit(EPS), logit(1 - EPS), 501))
        b_cnt = np.histogram(np.clip(uc, EPS, 1 - EPS), bins=b_edges)[0].astype(float)
        b_mid = expit(0.5 * (logit(b_edges[:-1]) + logit(b_edges[1:])))
        keep = b_cnt > 0
        b_cnt, b_mid = b_cnt[keep], b_mid[keep]
        for cname in ("beta_mle", "hist20"):
            ctp, cfp = comps[cname]

            def ll_delta(d):
                s2 = expit(logit(b_mid) - d)
                jac = np.log(s2 * (1 - s2)) - np.log(b_mid * (1 - b_mid))
                pi_d, ll_d, _ = em_pi(ctp.logpdf(s2), cfp.logpdf(s2), pi_src, w=b_cnt)
                return ll_d + float(np.sum(b_cnt * jac)), pi_d

            coarse = [ll_delta(d) for d in DELTA_GRID]
            j = int(np.argmax([c[0] for c in coarse]))
            fine = DELTA_GRID[j] + np.linspace(-0.1, 0.1, 21)
            fine_ll = [ll_delta(d) for d in fine]
            jf = int(np.argmax([c[0] for c in fine_ll]))
            delta, (ll3, pi3) = float(fine[jf]), fine_ll[jf]
            t05, tf1, f1hat, ngt_hat = mixture_thresholds(ctp, cfp, pi3, len(uc), r_src, float(sc.max()), delta=delta)
            thr[f"M3_{cname}_shift_post05"], thr[f"M3_{cname}_shift_mixf1"] = t05, tf1
            mix[f"{cname}_shift"] = {"delta_logit": delta, "pi_hat": pi3, "loglik": ll3, "t_post05": t05, "t_mixf1": tf1,
                                     "mixture_f1_hat": f1hat, "n_gt_hat": ngt_hat,
                                     "loglik_delta0": coarse[int(np.argmin(np.abs(DELTA_GRID)))][0],
                                     "delta_at_grid_edge": bool(abs(abs(delta) - DELTA_GRID[-1]) < 0.11)}
        thr["ORC_target_calval"] = t_orc
        thr["TOPT_test_optimal"] = t_topt

        # --- evaluate every threshold on the target test -----------------------------------
        test_dets_raw = None
        if not args.no_laece:
            test_dets_raw = [d for d in json.loads(Path(files["target_test"]).read_text(encoding="utf-8")) if d["category_id"] in cls_t]
        f1_orc = op.at_threshold(*X, t_orc)["f1"]
        res = {}
        for m, t in thr.items():
            b = op.at_threshold(*X, t)
            il = op.image_level(x_img, t)
            r = {"threshold": t, "box": b, "image": {"sensitivity": il["sensitivity"], "false_alarm_rate": il["false_alarm_rate"],
                                                     "alarm_precision": il["alarm_precision"]},
                 "regret_vs_oracle": f1_orc - b["f1"], "regret_vs_test_opt": float(X[3].max()) - b["f1"],
                 "abs_log10_ratio_to_oracle": abs(np.log10(max(t, 1e-6) / max(t_orc, 1e-6)))}
            if test_dets_raw is not None and m in LAECE_METHODS:
                kept = [d for d in test_dets_raw if d["score"] >= t]
                if len(kept) > LAECE_MAX_DETS:
                    r["LaECE_0"], r["LaECE_0_note"] = None, f"skipped: {len(kept)} kept detections > cap {LAECE_MAX_DETS}"
                else:
                    r["LaECE_0"] = laece0(str(GT / f"{tgt}.json"), kept)
            res[m] = r
        # --- shift diagnostics -----------------------------------------------------------------
        diag = {"wasserstein_pooled": float(stats.wasserstein_distance(sc, uc)) if len(uc) else None,
                "wasserstein_imgmax": float(stats.wasserstein_distance(s_max, u_max)),
                "f1_loss_at_t_src_vs_oracle": res["M0_t_src"]["regret_vs_oracle"],
                "abs_t_src_minus_t_oracle": abs(t_src - t_orc)}
        out["cells"][name] = {
            "run": run, "family": "rtdetr" if run.startswith("rtdetr") else "yolo",
            "source_calval": src, "target_calval_unlabelled": cal, "target_test_split": tgt, "files": files,
            "classes_evaluated": sorted(cls_t), "classes_source": sorted(cls_s),
            "n": {"source_dets": int(len(sc)), "source_gt": int(sng), "source_images": int(len(s_img)),
                  "target_calval_dets": int(len(uc)), "target_calval_images": int(n_u_img),
                  "target_test_dets": int(len(xc)), "target_test_gt": int(xng), "target_test_images": int(len(x_img))},
            "source": {"t_src": t_src, "t_src_operating_points": t_src_ref, "reproduces": abs(t_src - t_src_ref) < 1e-9,
                       "f1_at_t_src": float(S[3].max()), "quantile_pooled_at_t_src": q_pool, "quantile_imgmax_at_t_src": q_max,
                       "kept_dets_per_image_at_t_src": kept_rate, "tp_fraction_at_floor": pi_src, "gt_per_tp_at_floor": r_src},
            "target_calval_labels_for_evaluation_only": {"t_oracle": t_orc, "f1_at_oracle_on_calval": float(U[3].max()),
                                                          "tp_fraction_at_floor": float(utp.mean()) if len(utp) else None},
            "target_test": {"t_test_optimal": t_topt, "f1_test_optimal": float(X[3].max()), "tp_fraction_at_floor": float(xtp.mean()) if len(xtp) else None,
                            "f1_at_t_src_operating_points": opj["cells"][name]["target"]["box_at_t_src"]["f1"]},
            "mixture": mix, "shift_diagnostics": diag, "methods": res, "seconds": round(time.time() - t0, 1)}
        rows.append((name, res, diag))
        print(f"{name}: t_src {t_src:.3f} orc {t_orc:.3f} | M1a {thr['M1a_quantile_pooled']:.3f} M1b {thr['M1b_quantile_imgmax']:.3f} "
              f"M2mle05 {thr['M2_beta_mle_post05']:.3f} M2mleF1 {thr['M2_beta_mle_mixf1']:.3f} M3 {thr['M3_beta_mle_shift_post05']:.3f} "
              f"| F1 src {res['M0_t_src']['box']['f1']:.3f} orc {f1_orc:.3f} | pi_hat {mix['beta_mle']['pi_hat']:.3f} "
              f"true {out['cells'][name]['target_calval_labels_for_evaluation_only']['tp_fraction_at_floor']:.3f} | {time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------------- summary
    methods = list(rows[0][1].keys()) if rows else []
    def summarise(sel):
        s = {}
        for m in methods:
            reg = np.array([r[m]["regret_vs_oracle"] for _, r, _ in sel])
            f1 = np.array([r[m]["box"]["f1"] for _, r, _ in sel])
            f1_src = np.array([r["M0_t_src"]["box"]["f1"] for _, r, _ in sel])
            lr = np.array([r[m]["abs_log10_ratio_to_oracle"] for _, r, _ in sel])
            la = [r[m].get("LaECE_0") for _, r, _ in sel]
            la = np.array([v for v in la if v is not None])
            s[m] = {"n_cells": int(len(sel)), "regret_median": float(np.median(reg)), "regret_mean": float(reg.mean()),
                    "regret_max": float(reg.max()), "f1_median": float(np.median(f1)), "f1_mean": float(f1.mean()),
                    "n_beats_t_src": int(np.sum(f1 > f1_src + 1e-12)), "n_worse_than_t_src": int(np.sum(f1 < f1_src - 1e-12)),
                    "abs_log10_threshold_ratio_to_oracle_median": float(np.median(lr)),
                    "LaECE_0_median": float(np.median(la)) if len(la) else None, "LaECE_0_mean": float(la.mean()) if len(la) else None,
                    "fraction_of_gap_closed_mean": float(np.mean([(r[m]["box"]["f1"] - r["M0_t_src"]["box"]["f1"]) / g if g > 1e-9 else np.nan
                                                                   for _, r, _ in sel for g in [r["M0_t_src"]["regret_vs_oracle"]]
                                                                   if g > 1e-9])) if any(r["M0_t_src"]["regret_vs_oracle"] > 1e-9 for _, r, _ in sel) else None}
        return s

    groups = {"all": rows,
              "yolo": [r for r in rows if out["cells"][r[0]]["family"] == "yolo"],
              "rtdetr": [r for r in rows if out["cells"][r[0]]["family"] == "rtdetr"],
              "core_v8s_y11s_rtdetrl": [r for r in rows if out["cells"][r[0]]["run"].split("_")[0] in ("v8s", "y11s", "rtdetrl")
                                        and "dfiredd" not in r[0] and "dfiresub" not in r[0]]}
    for fam in ("yolo", "rtdetr"):
        for src_ds in ("d_fire", "pyro_sdis"):
            for tgt_ds in ("d_fire", "pyro_sdis", "thesis"):
                sel = [r for r in rows if out["cells"][r[0]]["family"] == fam
                       and dataset_of(out["cells"][r[0]]["source_calval"]) == src_ds and dataset_of(out["cells"][r[0]]["target_test_split"]) == tgt_ds]
                if sel:
                    groups[f"{fam}:{src_ds}->{tgt_ds}"] = sel
    summary = {g: summarise(sel) for g, sel in groups.items() if sel}
    corr = {}
    for key in ("wasserstein_pooled", "wasserstein_imgmax"):
        w = np.array([d[key] for _, _, d in rows], dtype=float)
        for tgt_key in ("f1_loss_at_t_src_vs_oracle", "abs_t_src_minus_t_oracle"):
            y = np.array([d[tgt_key] for _, _, d in rows], dtype=float)
            rho, p = stats.spearmanr(w, y)
            corr[f"{key}__vs__{tgt_key}"] = {"spearman_rho": float(rho), "p": float(p), "n": int(len(w))}
    pt = np.array([out["cells"][n]["target_calval_labels_for_evaluation_only"]["tp_fraction_at_floor"] for n in out["cells"]])
    for comp in ("beta_mle", "hist20", "beta_mle_shift", "hist20_shift"):
        ph = np.array([out["cells"][n]["mixture"][comp]["pi_hat"] for n in out["cells"]])
        corr[f"pi_hat_{comp}_vs_pi_true_calval"] = {"spearman_rho": float(stats.spearmanr(ph, pt)[0]) if len(ph) > 2 else None,
                                                    "mean_abs_error": float(np.mean(np.abs(ph - pt))), "mean_signed_error": float(np.mean(ph - pt)),
                                                    "median_ratio_hat_over_true": float(np.median(ph / np.maximum(pt, 1e-9)))}
    out["summary"] = {"groups": summary, "correlations": corr, "n_cells": len(rows), "seconds_total": round(time.time() - t_start, 1)}
    (HERE / "label_free_threshold_pilot.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------------- compact table + md
    lines = ["# Label-free threshold pilot — summary (machine-written by label_free_threshold_pilot.py)", "",
             f"Cells: {len(rows)} cross-dataset (skipped {len(skipped)}). Regret = F1(oracle: target-calval-labelled) - F1(method), box F1 at IoU 0.5 on the target test.", ""]
    for g, s in summary.items():
        lines += [f"## {g} (n = {list(s.values())[0]['n_cells']})", "",
                  "| method | regret median | regret mean | F1 median | beats t_src | worse | median |log10 t/t_orc| | LaECE_0 median | gap closed (mean) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for m, v in s.items():
            gc = v["fraction_of_gap_closed_mean"]
            lines.append(f"| {m} | {v['regret_median']:.3f} | {v['regret_mean']:.3f} | {v['f1_median']:.3f} | {v['n_beats_t_src']} | {v['n_worse_than_t_src']} | "
                         f"{v['abs_log10_threshold_ratio_to_oracle_median']:.3f} | {'' if v['LaECE_0_median'] is None else f'{v['LaECE_0_median']:.3f}'} | "
                         f"{'' if gc is None else f'{gc:.3f}'} |")
        lines.append("")
    lines += ["## Shift diagnostic correlations (Spearman across cells)", ""]
    for k, v in corr.items():
        lines.append(f"- {k}: " + ", ".join(f"{a} = {b:.3f}" if isinstance(b, float) else f"{a} = {b}" for a, b in v.items()))
    lines += ["", "## Per cell (threshold / F1 on target test)", "",
              "| cell | t_src | ORC | M1a | M1b | M1c | M2mle-post05 | M2mle-mixF1 | M2hist-post05 | M3mle-post05 | M3hist-post05 | F1 src | F1 M1a | F1 M1b | F1 M2mle05 | F1 M2mleF1 | F1 M3mle | F1 M3hist | F1 ORC | pi_hat M2mle/M3mle/true | delta M3mle |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r, _ in rows:
        c = out["cells"][n]
        f = lambda m, k="threshold": r[m][k] if k == "threshold" else r[m]["box"]["f1"]
        lines.append(f"| {n} | {f('M0_t_src'):.3f} | {f('ORC_target_calval'):.3f} | {f('M1a_quantile_pooled'):.3f} | {f('M1b_quantile_imgmax'):.3f} | {f('M1c_kept_rate'):.3f} | "
                     f"{f('M2_beta_mle_post05'):.3f} | {f('M2_beta_mle_mixf1'):.3f} | {f('M2_hist20_post05'):.3f} | {f('M3_beta_mle_shift_post05'):.3f} | {f('M3_hist20_shift_post05'):.3f} | "
                     f"{f('M0_t_src','f1'):.3f} | {f('M1a_quantile_pooled','f1'):.3f} | {f('M1b_quantile_imgmax','f1'):.3f} | {f('M2_beta_mle_post05','f1'):.3f} | {f('M2_beta_mle_mixf1','f1'):.3f} | "
                     f"{f('M3_beta_mle_shift_post05','f1'):.3f} | {f('M3_hist20_shift_post05','f1'):.3f} | {f('ORC_target_calval','f1'):.3f} | "
                     f"{c['mixture']['beta_mle']['pi_hat']:.3f}/{c['mixture']['beta_mle_shift']['pi_hat']:.3f}/{c['target_calval_labels_for_evaluation_only']['tp_fraction_at_floor']:.3f} | "
                     f"{c['mixture']['beta_mle_shift']['delta_logit']:+.2f} |")
    (HERE / "label_free_threshold_pilot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:60]))
    print("artefact ->", HERE / "label_free_threshold_pilot.json", "and .md", f"({out['summary']['seconds_total']} s)")


if __name__ == "__main__":
    main()
