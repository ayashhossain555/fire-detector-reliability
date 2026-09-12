"""Label-free threshold transfer — publication-grade refinement of the pilot (2026-09-12, CPU only).

Builds on label_free_threshold_pilot.py (cells, LaECE_0 call, matching conventions are
IMPORTED from it / operating_points.py unchanged). Same protocol per cross-dataset cell:
  source calval : detections + labels           -> t_src (source F1-optimal) and the source
                                                   kept-statistics at t_src
  target calval : detections, SCORES ONLY       -> adapted threshold (unlabelled adaptation set)
  target test   : detections + labels           -> evaluation (box F1 at IoU 0.5, image-level
                                                   sensitivity / false-alarm rate, LaECE_0)
  oracle (ORC)  : F1-optimal threshold on the target calval WITH labels (upper reference)

Estimators (label-free on the target; every estimator sees only the target-calval scores, the
image index of each detection and its within-image score rank):
  M0_t_src        ship the source threshold
  D25_default     0.25 (ultralytics default)
  M1c_kept_rate   kept-detections-per-image matching (pilot M1c): choose t so that the target
                  keeps the same mean number of detections per image as t_src keeps on the source
  M1c_top{k}      the same statistic computed from the top-k detections per image ONLY, on both
                  sides (k = 5, 10, 20, 50). Identity: mean_i min(rank_i(t), k), i.e. the per-image
                  rank of the last kept detection capped at k.
  M1d_rank_norm   match the mean per-image kept FRACTION rank_i(t) / n_dets_i (normalised rank)
  M1e_rank_log    match the mean per-image log(1 + rank_i(t)) (concave rank transform; damps
                  images that keep hundreds of low-score queries)
  M1r_rank_mean   match the mean per-image rank of the last kept detection = mean_i rank_i(t);
                  since sum_i rank_i(t) = #pooled detections >= t this is M1c by identity — kept
                  as a numerical check (reported max |t_M1r - t_M1c|).
All variants share one solver: sort the unlabelled target scores descending, cumulate the
per-detection increment of the statistic, and take the score at which the cumulated statistic is
nearest to (source statistic x n_target_images) (nearest attainable value; 0 kept -> t = 1.0).

Uncertainty: image-level bootstrap of the TARGET TEST set (B = 1000, seed 0, multinomial image
weights drawn once per target test split so that every cell on the same target shares the same
resamples). Thresholds are held fixed (they come from source / unlabelled target calval, not from
the test set), so the CIs describe evaluation-set sampling. Reported per cell: percentile 95 % CIs
for F1 at t_src / M1c / ORC / every variant, the paired differences M1c - t_src and ORC - M1c, the
fraction of the gap closed (M1c - t_src) / (ORC - t_src) (flagged unstable when the denominator is
near zero in many replicates), image-level sensitivity and false-alarm rate.
Group summaries (all / yolo / rtdetr, by model family, by target) carry two CIs: (i) image-bootstrap
propagated — mean over the group's cells of replicate-b regret; (ii) cell-level percentile bootstrap
(B = 1000, seed 0) — resampling cells; SD across cells also given.

LaECE_0 (toolbox, identity map, detections kept at the threshold, same call as the pilot; the
toolbox call itself uses maxDets = 100): computed for every cell and for t_src, D25, M1c, every
M1c_top{k}, M1d, M1e and ORC. Detections are PRE-FILTERED to the top 100 scores per image before
the threshold cut for every cell (RT-DETR exports 300 queries per image); the number of detections
removed by the pre-filter is recorded per cell (0 for every YOLO cell unless stated).

Unlabelled-data sensitivity: the target calval is subsampled to n = 25, 50, 100, 200, 500, 1000
images (10 draws each without replacement, seed 0 per cell; n > available images skipped), M1c and
every M1c_top{k} threshold recomputed from the subsample, regret on the full target test recorded
per draw.

Leakage check (asserted at run time, per cell): the target-calval thresholds are recomputed after
DELETING every target-calval annotation (matching re-run with an empty ground truth); every
estimator threshold must be bit-identical. Target-test labels are used only in evaluation; target-
calval labels only by ORC.
Writes analysis/label_free_threshold_refine.json (+ .md). CPU only (no torch import).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operating_points as op  # noqa: E402  (read-only reuse)
import label_free_threshold_pilot as pilot  # noqa: E402  (read-only reuse: select_cells, laece0, GT/DETS)

B = 1000
SEED = 0
TOPK = (5, 10, 20, 50)
SUBSAMPLE_N = (25, 50, 100, 200, 500, 1000)
N_DRAWS = 10
LAECE_TOP_PER_IMAGE = 100
MAIN = ["M0_t_src", "D25_default", "M1c_kept_rate"] + [f"M1c_top{k}" for k in TOPK] + ["M1d_rank_norm", "M1e_rank_log", "ORC_target_calval"]
LAECE_METHODS = set(MAIN)
VARIANTS = ["M1c_kept_rate", "M1r_rank_mean"] + [f"M1c_top{k}" for k in TOPK] + ["M1d_rank_norm", "M1e_rank_log"]
SUB_VARIANTS = ["M1c_kept_rate"] + [f"M1c_top{k}" for k in TOPK]


# ----------------------------------------------------------------------------- matching with image index
def match_indexed(img_ids, gt_boxes, dets_by_img, classes):
    """op.match run per image (unchanged logic) so that each detection carries its image index and
    its within-image score rank (1 = highest score in the image)."""
    S, T, I, R, per_img, n_gt = [], [], [], [], [], 0
    for i, iid in enumerate(img_ids):
        s, t, g, pi = op.match([iid], gt_boxes, dets_by_img, classes)
        n_gt += g
        per_img.append(pi[0])
        if len(s):
            S.append(s); T.append(t); I.append(np.full(len(s), i, dtype=np.int64)); R.append(np.arange(1, len(s) + 1, dtype=np.int64))
    cat = lambda L, dt: np.concatenate(L) if L else np.zeros(0, dtype=dt)
    d = {"score": cat(S, float), "tp": cat(T, bool), "img": cat(I, np.int64), "rank": cat(R, np.int64),
         "n_img": len(img_ids), "n_gt": int(n_gt),
         "img_max": np.array([m for m, _ in per_img], dtype=float), "img_pos": np.array([p for _, p in per_img], dtype=bool)}
    d["n_dets_img"] = np.bincount(d["img"], minlength=d["n_img"])
    d["gt_img"] = np.array([sum(1 for c, _ in gt_boxes.get(iid, []) if c in classes) for iid in img_ids], dtype=np.int64)
    assert int(d["gt_img"].sum()) == d["n_gt"]
    return d


def unlabelled_view(U):
    """The ONLY thing the estimators may see from the target calval."""
    return {"score": U["score"], "img": U["img"], "rank": U["rank"], "n_dets_img": U["n_dets_img"], "n_img": U["n_img"]}


# ----------------------------------------------------------------------------- estimators
def increment(view, variant):
    r = view["rank"]
    if variant in ("M1c_kept_rate", "M1r_rank_mean"):
        return np.ones(len(r), dtype=float)
    if variant.startswith("M1c_top"):
        return (r <= int(variant[len("M1c_top"):])).astype(float)
    if variant == "M1d_rank_norm":
        return 1.0 / view["n_dets_img"][view["img"]].astype(float)
    if variant == "M1e_rank_log":
        return np.log1p(r) - np.log(r)
    raise KeyError(variant)


def source_statistic(view, t_src, variant):
    """Mean over source images of the per-image statistic at t_src (only scores/ranks used)."""
    kept = view["score"] >= t_src
    return float(increment(view, variant)[kept].sum() / view["n_img"])


def solve_threshold(view, order, variant, stat_src):
    """Score at which the cumulated target statistic is nearest to stat_src * n_img (labels never seen)."""
    for forbidden in ("tp", "gt", "gt_img", "img_pos", "label"):
        assert forbidden not in view, f"leakage: estimator received '{forbidden}'"
    need = stat_src * view["n_img"]
    if len(order) == 0:
        return 1.0
    cum = np.cumsum(increment(view, variant)[order])
    if need < cum[0] / 2:
        return 1.0
    j = int(np.searchsorted(cum, need))
    cand = [i for i in (j - 1, j) if 0 <= i < len(cum)]
    j = min(cand, key=lambda i: (abs(cum[i] - need), i))
    return float(view["score"][order][j])


def all_thresholds(view, stats_src):
    order = np.argsort(-view["score"], kind="stable")
    return {v: solve_threshold(view, order, v, stats_src[v]) for v in VARIANTS}


# ----------------------------------------------------------------------------- evaluation
def counts_at(X, t):
    m = X["score"] >= t
    tp_i = np.bincount(X["img"][m], weights=X["tp"][m].astype(float), minlength=X["n_img"])
    nd_i = np.bincount(X["img"][m], minlength=X["n_img"]).astype(float)
    return tp_i, nd_i, (X["img_max"] >= t)


def f1_from(tp, nd, gt):
    den = nd + gt
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)


def evaluate(X, t, W):
    """Point estimates + bootstrap replicate vectors for one threshold on the target test."""
    tp_i, nd_i, alarm = counts_at(X, t)
    gt_i = X["gt_img"].astype(float)
    pos, neg = X["img_pos"].astype(float), (~X["img_pos"]).astype(float)
    point = {"f1": float(f1_from(tp_i.sum(), nd_i.sum(), gt_i.sum())),
             "precision": float(tp_i.sum() / nd_i.sum()) if nd_i.sum() > 0 else None,
             "recall": float(tp_i.sum() / gt_i.sum()) if gt_i.sum() > 0 else 0.0,
             "sensitivity": float((alarm & X["img_pos"]).sum() / pos.sum()) if pos.sum() else None,
             "false_alarm_rate": float((alarm & ~X["img_pos"]).sum() / neg.sum()) if neg.sum() else None}
    rep = {"f1": f1_from(W @ tp_i, W @ nd_i, W @ gt_i)}
    a = alarm.astype(float)
    wp, wn = W @ pos, W @ neg
    rep["sensitivity"] = np.where(wp > 0, (W @ (a * pos)) / np.maximum(wp, 1e-12), np.nan)
    rep["false_alarm_rate"] = np.where(wn > 0, (W @ (a * neg)) / np.maximum(wn, 1e-12), np.nan)
    return point, rep


def ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return None
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def stat_block(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return None
    return {"mean": float(x.mean()), "median": float(np.median(x)), "ci95": ci(x), "n": int(len(x))}


def top_per_image(raw, n):
    """Keep the n highest-scoring raw detection dicts per image."""
    by = {}
    for d in raw:
        by.setdefault(d["image_id"], []).append(d)
    out = []
    for v in by.values():
        v.sort(key=lambda d: -d["score"])
        out.extend(v[:n])
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter on cell name (debug)")
    ap.add_argument("--no-laece", action="store_true")
    ap.add_argument("--no-subsample", action="store_true")
    ap.add_argument("--B", type=int, default=B)
    args = ap.parse_args()
    B_ = args.B
    t_start = time.time()
    opj = json.loads((HERE / "operating_points.json").read_text(encoding="utf-8"))
    pj = json.loads((HERE / "label_free_threshold_pilot.json").read_text(encoding="utf-8"))
    cells, skipped = pilot.select_cells(opj)
    if args.only:
        cells = [c for c in cells if args.only in c[0]]
    print(f"{len(cells)} cells, {len(skipped)} skipped; B = {B_}, seed = {SEED}", flush=True)

    gt_cache, det_cache, W_cache = {}, {}, {}

    def gt(tag):
        if tag not in gt_cache:
            gt_cache[tag] = op.load_gt(tag)
        return gt_cache[tag]

    def dets(run, tag):
        k = (run, tag)
        if k not in det_cache:
            det_cache[k] = op.load_dets(pilot.DETS / f"{run}__{tag}.bbox.json")
        return det_cache[k]

    def weights(tag, n_img):
        if tag not in W_cache:
            rng = np.random.default_rng(SEED)
            W_cache[tag] = rng.multinomial(n_img, np.full(n_img, 1.0 / n_img), size=B_).astype(float)
        assert W_cache[tag].shape == (B_, n_img)
        return W_cache[tag]

    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "iou": op.IOU, "bootstrap": {"B": B_, "seed": SEED, "unit": "target-test image (multinomial weights)", "ci": "percentile 2.5-97.5"},
           "topk": list(TOPK), "subsample": {"n": list(SUBSAMPLE_N), "draws": N_DRAWS, "seed": SEED, "without_replacement": True},
           "laece_prefilter_top_per_image": LAECE_TOP_PER_IMAGE, "laece_toolbox_maxDets": 100,
           "protocol": __doc__,
           "leakage_check": {"rule": "estimators receive only {score, img, rank, n_dets_img, n_img} of the target calval; thresholds recomputed with every target-calval annotation deleted must be identical; asserted per cell", "cells_checked": 0, "max_abs_threshold_diff": 0.0, "passed": None},
           "skipped": skipped, "cells": {}}
    reps = {}          # cell -> method -> replicate arrays
    rows = []
    max_leak_diff, n_leak = 0.0, 0
    max_m1r_diff, n_m1c_pilot_mismatch, max_m1c_pilot_diff = 0.0, 0, 0.0
    for name, run, src, cal, tgt, files in cells:
        t0 = time.time()
        s_ids, s_gt, s_cls = gt(src)
        u_ids, u_gt, _ = gt(cal)
        x_ids, x_gt, t_cls = gt(tgt)
        s_d, u_d, x_d = dets(run, src), dets(run, cal), dets(run, tgt)
        model_cls = {c for v in s_d.values() for c, _, _ in v} | {c for v in x_d.values() for c, _, _ in v}
        cls_s, cls_t = s_cls & model_cls, t_cls & model_cls
        # --- source calval (labelled) ------------------------------------------------------------
        S = match_indexed(s_ids, s_gt, s_d, cls_s)
        Sc = op.pr_curve(S["score"], S["tp"], S["n_gt"])
        t_src = float(Sc[0][int(np.argmax(Sc[3]))])
        assert abs(t_src - opj["cells"][name]["source"]["t_f1opt"]) < 1e-9
        stats_src = {v: source_statistic(unlabelled_view(S), t_src, v) for v in VARIANTS}
        # --- target calval: scores only for the estimators; labels ONLY for ORC ----------------------
        U = match_indexed(u_ids, u_gt, u_d, cls_t)
        view = unlabelled_view(U)
        thr = {"M0_t_src": t_src, "D25_default": 0.25}
        thr.update(all_thresholds(view, stats_src))
        Uc = op.pr_curve(U["score"], U["tp"], U["n_gt"])
        t_orc = float(Uc[0][int(np.argmax(Uc[3]))])
        thr["ORC_target_calval"] = t_orc
        # leakage assertion: delete every target-calval annotation and recompute
        U0 = match_indexed(u_ids, {}, u_d, cls_t)
        thr0 = all_thresholds(unlabelled_view(U0), stats_src)
        d_leak = max(abs(thr[v] - thr0[v]) for v in VARIANTS)
        assert d_leak == 0.0, f"{name}: thresholds depend on target-calval labels (diff {d_leak})"
        max_leak_diff, n_leak = max(max_leak_diff, d_leak), n_leak + 1
        max_m1r_diff = max(max_m1r_diff, abs(thr["M1r_rank_mean"] - thr["M1c_kept_rate"]))
        pm = pj["cells"][name]["methods"]
        dm = abs(thr["M1c_kept_rate"] - pm["M1c_kept_rate"]["threshold"])
        max_m1c_pilot_diff = max(max_m1c_pilot_diff, dm)
        n_m1c_pilot_mismatch += int(dm > 1e-12)
        # --- target test (labelled evaluation) + bootstrap -------------------------------------------
        X = match_indexed(x_ids, x_gt, x_d, cls_t)
        W = weights(tgt, X["n_img"])
        Xc = op.pr_curve(X["score"], X["tp"], X["n_gt"])
        f1_topt = float(Xc[3].max())
        res, rep = {}, {}
        for m, t in thr.items():
            point, r = evaluate(X, t, W)
            res[m], rep[m] = {"threshold": t, **point}, r
        # exact reproduction of the pilot point estimates (same matching, same F1 definition)
        for m in ("M0_t_src", "M1c_kept_rate", "ORC_target_calval", "D25_default"):
            assert abs(res[m]["f1"] - op.at_threshold(*Xc, thr[m])["f1"]) < 1e-9, (name, m)
            if m in pm and abs(thr[m] - pm[m]["threshold"]) < 1e-12:
                assert abs(res[m]["f1"] - pm[m]["box"]["f1"]) < 1e-9, (name, m, res[m]["f1"], pm[m]["box"]["f1"])
        f1_orc = res["ORC_target_calval"]["f1"]
        f1_src = res["M0_t_src"]["f1"]
        gap_point = f1_orc - f1_src
        gap_rep = rep["ORC_target_calval"]["f1"] - rep["M0_t_src"]["f1"]
        for m in thr:
            r = res[m]
            r["regret_vs_oracle"] = f1_orc - r["f1"]
            r["regret_vs_test_opt"] = f1_topt - r["f1"]
            r["abs_log10_ratio_to_oracle"] = abs(float(np.log10(max(thr[m], 1e-6) / max(t_orc, 1e-6))))
            r["ci95"] = {k: ci(rep[m][k]) for k in ("f1", "sensitivity", "false_alarm_rate")}
            r["regret_vs_oracle_ci95"] = ci(rep["ORC_target_calval"]["f1"] - rep[m]["f1"])
            if m != "M0_t_src":
                d = rep[m]["f1"] - rep["M0_t_src"]["f1"]
                r["diff_vs_t_src"] = {"f1": r["f1"] - f1_src, "ci95": ci(d), "p_boot_le_0": float(np.mean(d <= 0))}
                rep[m]["diff_vs_t_src"] = d
                valid = gap_rep > 0.005
                gc = np.where(valid, d / np.where(valid, gap_rep, 1.0), np.nan)
                r["gap_closed"] = {"point": (r["f1"] - f1_src) / gap_point if gap_point > 1e-9 else None,
                                   "ci95": ci(gc), "n_valid_replicates": int(valid.sum()),
                                   "unstable": bool(valid.sum() < 0.95 * B_)}
                rep[m]["gap_closed"] = gc
                for k in ("sensitivity", "false_alarm_rate"):
                    r["diff_vs_t_src"][k] = {"point": (r[k] - res["M0_t_src"][k]) if (r[k] is not None and res["M0_t_src"][k] is not None) else None,
                                             "ci95": ci(rep[m][k] - rep["M0_t_src"][k])}
            rep[m]["regret"] = rep["ORC_target_calval"]["f1"] - rep[m]["f1"]
        res["ORC_target_calval"]["gap_ci95"] = ci(gap_rep)
        # --- LaECE_0 --------------------------------------------------------------------------------
        laece_note = None
        if not args.no_laece:
            raw = [d for d in json.loads(Path(files["target_test"]).read_text(encoding="utf-8")) if d["category_id"] in cls_t]
            raw_f = top_per_image(raw, LAECE_TOP_PER_IMAGE)
            laece_note = {"n_dets": len(raw), "n_after_prefilter": len(raw_f), "n_removed_by_prefilter": len(raw) - len(raw_f)}
            for m in MAIN:
                kept = [d for d in raw_f if d["score"] >= thr[m]]
                res[m]["LaECE_0"] = pilot._q(pilot.laece0, str(pilot.GT / f"{tgt}.json"), kept)
                res[m]["LaECE_0_n_kept"] = len(kept)
        # --- unlabelled-data sensitivity -------------------------------------------------------------
        sub = {}
        if not args.no_subsample:
            rng = np.random.default_rng(SEED)
            img_of_det = U["img"]
            for n in SUBSAMPLE_N:
                if n > U["n_img"]:
                    sub[str(n)] = {"skipped": f"n > target calval images ({U['n_img']})"}
                    continue
                per_v = {v: {"regret": [], "threshold": []} for v in SUB_VARIANTS}
                for _ in range(N_DRAWS):
                    pick = rng.choice(U["n_img"], n, replace=False)
                    keep_img = np.zeros(U["n_img"], dtype=bool); keep_img[pick] = True
                    m = keep_img[img_of_det]
                    sv = {"score": U["score"][m], "img": U["img"][m], "rank": U["rank"][m], "n_dets_img": U["n_dets_img"], "n_img": n}
                    order = np.argsort(-sv["score"], kind="stable")
                    for v in SUB_VARIANTS:
                        t = solve_threshold(sv, order, v, stats_src[v])
                        tp_i, nd_i, _ = counts_at(X, t)
                        f1 = float(f1_from(tp_i.sum(), nd_i.sum(), X["n_gt"]))
                        per_v[v]["regret"].append(f1_orc - f1); per_v[v]["threshold"].append(t)
                sub[str(n)] = {v: {"regret_mean": float(np.mean(d["regret"])), "regret_median": float(np.median(d["regret"])),
                                   "regret_max": float(np.max(d["regret"])), "regret_draws": [float(x) for x in d["regret"]],
                                   "threshold_draws": [float(x) for x in d["threshold"]],
                                   "abs_log10_ratio_to_oracle_mean": float(np.mean([abs(np.log10(max(t, 1e-6) / max(t_orc, 1e-6))) for t in d["threshold"]]))}
                               for v, d in per_v.items()}
        # --- record ----------------------------------------------------------------------------------
        fam = run.split("_")[0]
        cell = {"run": run, "family": "rtdetr" if run.startswith("rtdetr") else "yolo", "model": fam,
                "train_set": run.split("_")[1], "seed": run.split("_")[2] if len(run.split("_")) > 2 else None,
                "source_calval": src, "target_calval_unlabelled": cal, "target_test": tgt, "target": pilot.dataset_of(tgt), "files": files,
                "classes_evaluated": sorted(cls_t), "classes_source": sorted(cls_s),
                "n": {"source_dets": int(len(S["score"])), "source_gt": S["n_gt"], "source_images": S["n_img"],
                      "target_calval_dets": int(len(U["score"])), "target_calval_images": U["n_img"],
                      "target_test_dets": int(len(X["score"])), "target_test_gt": X["n_gt"], "target_test_images": X["n_img"],
                      "max_dets_per_image_source": int(S["n_dets_img"].max()), "max_dets_per_image_target_calval": int(U["n_dets_img"].max())},
                "source": {"t_src": t_src, "f1_at_t_src": float(Sc[3].max()), "statistics_at_t_src": stats_src},
                "target_calval_labels_for_oracle_only": {"t_oracle": t_orc, "f1_at_oracle_on_calval": float(Uc[3].max())},
                "target_test_summary": {"t_test_optimal": float(Xc[0][int(np.argmax(Xc[3]))]), "f1_test_optimal": f1_topt,
                                "gap_oracle_minus_t_src": gap_point, "gap_ci95": res["ORC_target_calval"]["gap_ci95"]},
                "leakage_check": {"max_abs_threshold_diff_after_deleting_target_calval_labels": d_leak, "passed": d_leak == 0.0},
                "pilot_reproduction": {"M1c_threshold_abs_diff": dm, "M1r_rank_mean_vs_M1c_abs_diff": abs(thr["M1r_rank_mean"] - thr["M1c_kept_rate"])},
                "laece": laece_note, "methods": res, "unlabelled_data_sensitivity": sub, "seconds": round(time.time() - t0, 1)}
        out["cells"][name] = cell
        reps[name] = rep
        rows.append(name)
        print(f"{name}: t_src {t_src:.3f} orc {t_orc:.3f} M1c {thr['M1c_kept_rate']:.3f} top5 {thr['M1c_top5']:.3f} top20 {thr['M1c_top20']:.3f} "
              f"log {thr['M1e_rank_log']:.3f} | F1 src {f1_src:.3f} [{res['M0_t_src']['ci95']['f1'][0]:.3f},{res['M0_t_src']['ci95']['f1'][1]:.3f}] "
              f"M1c {res['M1c_kept_rate']['f1']:.3f} top20 {res['M1c_top20']['f1']:.3f} orc {f1_orc:.3f} | {time.time() - t0:.0f}s", flush=True)

    out["leakage_check"].update({"cells_checked": n_leak, "max_abs_threshold_diff": max_leak_diff, "passed": bool(n_leak == len(rows) and max_leak_diff == 0.0)})
    out["pilot_reproduction"] = {"M1c_threshold_max_abs_diff": max_m1c_pilot_diff, "M1c_threshold_n_cells_differing": n_m1c_pilot_mismatch,
                                 "M1r_rank_mean_vs_M1c_max_abs_diff": max_m1r_diff}

    # ---------------------------------------------------------------------- summaries
    methods = list(next(iter(reps.values())).keys()) if rows else []
    C = out["cells"]

    def group_summary(sel, rng_cells):
        s = {}
        for m in methods:
            reg = np.array([C[c]["methods"][m]["regret_vs_oracle"] for c in sel])
            f1 = np.array([C[c]["methods"][m]["f1"] for c in sel])
            f1s = np.array([C[c]["methods"]["M0_t_src"]["f1"] for c in sel])
            reg_rep = np.mean([reps[c][m]["regret"] for c in sel], axis=0)
            la = np.array([v for v in (C[c]["methods"][m].get("LaECE_0") for c in sel) if v is not None])
            gcs = [C[c]["methods"][m]["gap_closed"]["point"] for c in sel if m != "M0_t_src"]
            gcs = np.array([g for g in gcs if g is not None], dtype=float)
            cb = [reg[rng_cells.integers(0, len(sel), len(sel))].mean() for _ in range(B_)] if len(sel) >= 2 else None
            e = {"n_cells": int(len(sel)), "regret_mean": float(reg.mean()), "regret_median": float(np.median(reg)), "regret_max": float(reg.max()),
                 "regret_sd_across_cells": float(reg.std(ddof=1)) if len(sel) > 1 else None,
                 "regret_mean_ci95_image_bootstrap": ci(reg_rep), "regret_mean_ci95_cell_bootstrap": ci(cb) if cb is not None else None,
                 "f1_mean": float(f1.mean()), "n_beats_t_src": int(np.sum(f1 > f1s + 1e-12)), "n_worse_than_t_src": int(np.sum(f1 < f1s - 1e-12)),
                 "LaECE_0_median": float(np.median(la)) if len(la) else None, "LaECE_0_mean": float(la.mean()) if len(la) else None,
                 "fraction_of_gap_closed_mean": float(gcs.mean()) if len(gcs) else None,
                 "fraction_of_gap_closed_median": float(np.median(gcs)) if len(gcs) else None}
            if m != "M0_t_src":
                d = np.array([C[c]["methods"][m]["diff_vs_t_src"]["f1"] for c in sel])
                d_rep = np.mean([reps[c][m]["diff_vs_t_src"] for c in sel], axis=0)
                e["f1_gain_vs_t_src_mean"] = float(d.mean()); e["f1_gain_vs_t_src_ci95_image_bootstrap"] = ci(d_rep)
                e["n_cells_gain_ci_excludes_0"] = int(sum(1 for c in sel if C[c]["methods"][m]["diff_vs_t_src"]["ci95"][0] > 0))
                e["n_cells_loss_ci_excludes_0"] = int(sum(1 for c in sel if C[c]["methods"][m]["diff_vs_t_src"]["ci95"][1] < 0))
            for k in ("sensitivity", "false_alarm_rate"):
                pts = np.array([v for v in (C[c]["methods"][m][k] for c in sel) if v is not None], dtype=float)
                e[f"{k}_mean"] = float(pts.mean()) if len(pts) else None
                e[f"{k}_mean_ci95_image_bootstrap"] = ci(np.nanmean([reps[c][m][k] for c in sel], axis=0))
                if m != "M0_t_src":
                    dp = np.array([v for v in (C[c]["methods"][m]["diff_vs_t_src"][k]["point"] for c in sel) if v is not None], dtype=float)
                    e[f"{k}_diff_vs_t_src_mean"] = float(dp.mean()) if len(dp) else None
                    e[f"{k}_diff_vs_t_src_ci95_image_bootstrap"] = ci(np.nanmean([reps[c][m][k] - reps[c]["M0_t_src"][k] for c in sel], axis=0))
                    e[f"{k}_n_cells_diff_ci_excludes_0"] = int(sum(1 for c in sel if (lambda q: q is not None and (q[0] > 0 or q[1] < 0))(C[c]["methods"][m]["diff_vs_t_src"][k]["ci95"])))
            if m.startswith("M1") and m != "M1c_kept_rate":
                dd = np.array([C[c]["methods"][m]["f1"] - C[c]["methods"]["M1c_kept_rate"]["f1"] for c in sel])
                dd_rep = np.mean([reps[c][m]["f1"] - reps[c]["M1c_kept_rate"]["f1"] for c in sel], axis=0)
                e["f1_vs_M1c_mean"] = float(dd.mean()); e["f1_vs_M1c_ci95_image_bootstrap"] = ci(dd_rep)
                e["n_cells_better_than_M1c"] = int(np.sum(dd > 1e-12)); e["n_cells_worse_than_M1c"] = int(np.sum(dd < -1e-12))
            s[m] = e
        return s

    rng_cells = np.random.default_rng(SEED)
    groups = {"all": rows, "yolo": [c for c in rows if C[c]["family"] == "yolo"], "rtdetr": [c for c in rows if C[c]["family"] == "rtdetr"]}
    summary = {g: group_summary(sel, rng_cells) for g, sel in groups.items() if sel}
    by_model = {}
    for fam in sorted({C[c]["model"] for c in rows}):
        by_model[fam] = group_summary([c for c in rows if C[c]["model"] == fam], rng_cells)
    by_target = {}
    for tg in sorted({C[c]["target_test"] for c in rows}):
        by_target[tg] = group_summary([c for c in rows if C[c]["target_test"] == tg], rng_cells)
    # best k for RT-DETR
    topk_view = {}
    for k in TOPK:
        m = f"M1c_top{k}"
        topk_view[str(k)] = {g: {"regret_mean": summary[g][m]["regret_mean"], "regret_median": summary[g][m]["regret_median"],
                                 "regret_mean_ci95_image_bootstrap": summary[g][m]["regret_mean_ci95_image_bootstrap"],
                                 "f1_vs_M1c_mean": summary[g][m]["f1_vs_M1c_mean"], "f1_vs_M1c_ci95": summary[g][m]["f1_vs_M1c_ci95_image_bootstrap"],
                                 "n_better_than_M1c": summary[g][m]["n_cells_better_than_M1c"], "n_worse_than_M1c": summary[g][m]["n_cells_worse_than_M1c"]}
                             for g in summary}
    best_k, topk_note = None, None
    if "rtdetr" in summary:
        best_k = min(TOPK, key=lambda k: (summary["rtdetr"][f"M1c_top{k}"]["regret_mean"], -k))
        ident = [k for k in TOPK if summary["all"][f"M1c_top{k}"]["n_cells_better_than_M1c"] + summary["all"][f"M1c_top{k}"]["n_cells_worse_than_M1c"] == 0]
        topk_note = (f"best k by RT-DETR mean regret = {best_k}; k in {ident} give thresholds identical to M1c on every cell "
                     f"(the detections kept at t_src / at the matched target threshold already lie within the top-{min(ident) if ident else 'n/a'} ranks per image), "
                     f"so no top-k restriction changes RT-DETR; smaller k only removes information.")
    # data-size curve (aggregate over cells: per n, the mean over cells of the per-cell mean regret, and the pooled draw distribution)
    curve = {}
    for n in SUBSAMPLE_N:
        curve[str(n)] = {}
        for v in SUB_VARIANTS:
            for g, sel in groups.items():
                draws = [r for c in sel for r in (C[c]["unlabelled_data_sensitivity"].get(str(n), {}).get(v, {}).get("regret_draws") or [])]
                cells_n = [c for c in sel if v in C[c]["unlabelled_data_sensitivity"].get(str(n), {})]
                if not draws:
                    continue
                d = np.array(draws)
                full = np.array([C[c]["methods"][v]["regret_vs_oracle"] for c in cells_n])
                curve[str(n)].setdefault(v, {})[g] = {
                    "n_cells": len(cells_n), "n_draws": int(len(d)), "regret_mean": float(d.mean()), "regret_median": float(np.median(d)),
                    "regret_p90": float(np.percentile(d, 90)), "regret_max": float(d.max()),
                    "frac_draws_regret_le_0.01": float(np.mean(d <= 0.01)), "frac_draws_regret_le_0.02": float(np.mean(d <= 0.02)),
                    "full_calval_regret_mean_same_cells": float(full.mean()),
                    "excess_over_full_calval_mean": float(d.mean() - full.mean())}
    out["summary"] = {"groups": summary, "by_model": by_model, "by_target": by_target, "topk": topk_view, "best_k_rtdetr_by_mean_regret": best_k, "topk_note": topk_note,
                      "unlabelled_data_curve": curve, "n_cells": len(rows), "seconds_total": round(time.time() - t_start, 1)}
    (HERE / "label_free_threshold_refine.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------------- md
    def f3(x):
        return "" if x is None else f"{x:.3f}"

    def cis(c):
        return "" if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]"

    L = ["# Label-free threshold transfer — refinement summary (machine-written by label_free_threshold_refine.py)", "",
         f"Cells: {len(rows)} cross-dataset. Regret = F1(oracle) - F1(method), box F1 at IoU 0.5 on the target test. "
         f"CIs: image-level bootstrap of the target test, B = {B_}, seed {SEED}, percentile 95 %. LaECE_0: toolbox, identity map, "
         f"detections pre-filtered to top {LAECE_TOP_PER_IMAGE}/image. Leakage check passed: {out['leakage_check']['passed']} "
         f"({out['leakage_check']['cells_checked']} cells). Pilot M1c reproduced: max |dt| = {max_m1c_pilot_diff:.2e} "
         f"({n_m1c_pilot_mismatch} cells differ). M1r (mean rank) vs M1c: max |dt| = {max_m1r_diff:.2e}.", ""]
    for g, s in summary.items():
        L += [f"## {g} (n = {s['M0_t_src']['n_cells']})", "",
              "| method | regret mean | CI (image boot) | CI (cell boot) | regret median | gain vs t_src mean | CI | beats t_src | worse | gain CI>0 | loss CI<0 | gap closed mean | vs M1c mean | CI | LaECE_0 median |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, v in s.items():
            L.append(f"| {m} | {f3(v['regret_mean'])} | {cis(v['regret_mean_ci95_image_bootstrap'])} | {cis(v['regret_mean_ci95_cell_bootstrap'])} | {f3(v['regret_median'])} | "
                     f"{f3(v.get('f1_gain_vs_t_src_mean'))} | {cis(v.get('f1_gain_vs_t_src_ci95_image_bootstrap'))} | {v['n_beats_t_src']} | {v['n_worse_than_t_src']} | "
                     f"{v.get('n_cells_gain_ci_excludes_0', '')} | {v.get('n_cells_loss_ci_excludes_0', '')} | {f3(v['fraction_of_gap_closed_mean'])} | "
                     f"{f3(v.get('f1_vs_M1c_mean'))} | {cis(v.get('f1_vs_M1c_ci95_image_bootstrap'))} | {f3(v['LaECE_0_median'])} |")
        L.append("")
    L += ["## Image-level alarm metrics (mean over cells; image-bootstrap CI of the mean)", "",
          "| group | method | sensitivity | CI | FAR | CI | d sens vs t_src | CI | d FAR vs t_src | CI |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g, s in summary.items():
        for m in ("M0_t_src", "D25_default", "M1c_kept_rate", "ORC_target_calval"):
            v = s[m]
            L.append(f"| {g} | {m} | {f3(v['sensitivity_mean'])} | {cis(v['sensitivity_mean_ci95_image_bootstrap'])} | {f3(v['false_alarm_rate_mean'])} | {cis(v['false_alarm_rate_mean_ci95_image_bootstrap'])} | "
                     f"{f3(v.get('sensitivity_diff_vs_t_src_mean'))} | {cis(v.get('sensitivity_diff_vs_t_src_ci95_image_bootstrap'))} | {f3(v.get('false_alarm_rate_diff_vs_t_src_mean'))} | {cis(v.get('false_alarm_rate_diff_vs_t_src_ci95_image_bootstrap'))} |")
    L += ["", f"## Top-k variants (best k for RT-DETR by mean regret: {best_k}) — {topk_note}", "", "| k | group | regret mean | CI | F1 vs M1c mean | CI | better/worse than M1c |", "|---|---|---|---|---|---|---|"]
    for k, gv in topk_view.items():
        for g, v in gv.items():
            L.append(f"| {k} | {g} | {f3(v['regret_mean'])} | {cis(v['regret_mean_ci95_image_bootstrap'])} | {f3(v['f1_vs_M1c_mean'])} | {cis(v['f1_vs_M1c_ci95'])} | {v['n_better_than_M1c']}/{v['n_worse_than_M1c']} |")
    L += ["", "## Unlabelled target data needed (regret over 10 draws x cells; full = same cells at the full calval)", "",
          "| n images | variant | group | cells | regret mean | median | p90 | max | frac <= 0.01 | frac <= 0.02 | full-calval mean | excess |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, vv in curve.items():
        for v, gg in vv.items():
            for g, e in gg.items():
                L.append(f"| {n} | {v} | {g} | {e['n_cells']} | {f3(e['regret_mean'])} | {f3(e['regret_median'])} | {f3(e['regret_p90'])} | {f3(e['regret_max'])} | "
                         f"{f3(e['frac_draws_regret_le_0.01'])} | {f3(e['frac_draws_regret_le_0.02'])} | {f3(e['full_calval_regret_mean_same_cells'])} | {f3(e['excess_over_full_calval_mean'])} |")
    for title, grp in (("By model", by_model), ("By target", by_target)):
        L += ["", f"## {title} (regret mean, image-bootstrap CI; cell-bootstrap CI; SD across cells)", "",
              "| group | n | t_src | M1c | CI img | CI cell | SD | best top-k | M1e log | ORC gap (t_src regret) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for g, s in grp.items():
            bk = s[f"M1c_top{best_k}"]["regret_mean"] if best_k else None
            L.append(f"| {g} | {s['M0_t_src']['n_cells']} | {f3(s['M0_t_src']['regret_mean'])} | {f3(s['M1c_kept_rate']['regret_mean'])} | "
                     f"{cis(s['M1c_kept_rate']['regret_mean_ci95_image_bootstrap'])} | {cis(s['M1c_kept_rate']['regret_mean_ci95_cell_bootstrap'])} | "
                     f"{f3(s['M1c_kept_rate']['regret_sd_across_cells'])} | {f3(bk)} | {f3(s['M1e_rank_log']['regret_mean'])} | {f3(s['M0_t_src']['regret_mean'])} |")
    L += ["", "## Per cell (F1 on target test with 95 % CI; regret; LaECE_0)", "",
          "| cell | t_src | t_M1c | t_ORC | F1 t_src [CI] | F1 M1c [CI] | F1 ORC [CI] | M1c-t_src [CI] | ORC-M1c [CI] | gap closed [CI] | F1 best top-k | LaECE t_src / D25 / M1c / top-k / ORC |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in rows:
        r = C[c]["methods"]
        bk = f"M1c_top{best_k}" if best_k else "M1c_kept_rate"
        gc = r["M1c_kept_rate"]["gap_closed"]
        L.append(f"| {c} | {f3(r['M0_t_src']['threshold'])} | {f3(r['M1c_kept_rate']['threshold'])} | {f3(r['ORC_target_calval']['threshold'])} | "
                 f"{f3(r['M0_t_src']['f1'])} {cis(r['M0_t_src']['ci95']['f1'])} | {f3(r['M1c_kept_rate']['f1'])} {cis(r['M1c_kept_rate']['ci95']['f1'])} | "
                 f"{f3(r['ORC_target_calval']['f1'])} {cis(r['ORC_target_calval']['ci95']['f1'])} | "
                 f"{f3(r['M1c_kept_rate']['diff_vs_t_src']['f1'])} {cis(r['M1c_kept_rate']['diff_vs_t_src']['ci95'])} | "
                 f"{f3(r['M1c_kept_rate']['regret_vs_oracle'])} {cis(r['M1c_kept_rate']['regret_vs_oracle_ci95'])} | "
                 f"{f3(gc['point'])} {cis(gc['ci95'])}{' (unstable)' if gc['unstable'] else ''} | {f3(r[bk]['f1'])} | "
                 f"{f3(r['M0_t_src'].get('LaECE_0'))} / {f3(r['D25_default'].get('LaECE_0'))} / {f3(r['M1c_kept_rate'].get('LaECE_0'))} / {f3(r[bk].get('LaECE_0'))} / {f3(r['ORC_target_calval'].get('LaECE_0'))} |")
    (HERE / "label_free_threshold_refine.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))
    print("artefact ->", HERE / "label_free_threshold_refine.json", "and .md", f"({out['summary']['seconds_total']} s)")


if __name__ == "__main__":
    main()
