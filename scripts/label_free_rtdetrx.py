"""Label-free threshold transfer for the two RT-DETR-x runs (2026-09-14, CPU only).

rtdetrx_dfire_s3407 and rtdetrx_pyrosdis_s3407 finished after the frozen 44-cell refine artefact
(label_free_threshold_refine.json, 22 runs). This script evaluates their four labelled cross cells
    rtdetrx_dfire_s3407    -> pyro_sdis_caltest, thesis_test
    rtdetrx_pyrosdis_s3407 -> d_fire_test_smokeonly, thesis_test_smokeonly
with EXACTLY the refine protocol, reusing its functions unchanged (read-only imports of
label_free_threshold_refine.py, label_free_threshold_pilot.py, operating_points.py):
  t_src        F1-optimal threshold on the labelled source calval (asserted equal to operating_points.json)
  kappa        the source kept-rate statistic at t_src = mean kept detections per source image
               (refine.source_statistic, variant M1c_kept_rate)
  M1c_full     kept-rate rule on the FULL unlabelled target calval (refine.solve_threshold)
  M1c_100      kept-rate rule on the seed-0 100-frame draw of the target calval: numpy default_rng(0),
               the refine data-size-curve draw sequence replayed (10 draws at n = 25, 10 at n = 50,
               then the FIRST draw at n = 100) exactly as bn_adaptation_baseline.replay_draw
  M1a          pooled-quantile matching F_tgt^-1(F_src(t_src)) (pilot M1a: np.quantile of the pooled
               target-calval scores at the source quantile of t_src)
  D25          0.25 (ultralytics default)
  ORC          F1-optimal threshold on the target calval WITH labels (labelled oracle)
Evaluation: box F1 at IoU 0.5 on the target test (op.match / op.pr_curve), image-level sensitivity
and false-alarm rate, regret = F1(ORC) - F1(method), image-level bootstrap of the target test
(B = 1000, seed 0, multinomial weights drawn once per target split -- identical to the refine),
LaECE_0 of the identity map on the detections kept (top-100 per image pre-filter, toolbox call as the
refine), W1 per-image-max shift diagnostic (scipy wasserstein_distance, source calval vs target
calval, as the pilot) with the in-domain null cut-off of label_free_clean_check.json.
Leakage check: thresholds recomputed with every target-calval annotation deleted must be identical.

Verification: the same code path is run on the four RT-DETR-l seed-3407 cells of the same directions
and asserted against the refine artefact (t_M1c, t_ORC to 1e-6; F1 at t_src / D25 / M1c / ORC to
1e-9; regret CI of M1c; LaECE_0; the n = 100 draw-0 threshold and regret), against the pilot
artefact (t_M1a, F1 at M1a, W1) and against operating_points.json (t_src).
Writes analysis/label_free_rtdetrx.json (+ .md). Does not modify any existing file.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operating_points as op  # noqa: E402  (read-only reuse)
import label_free_threshold_pilot as pilot  # noqa: E402  (read-only reuse)
import label_free_threshold_refine as refine  # noqa: E402  (read-only reuse)

B, SEED, N_ADAPT = 1000, 0, 100
NEW_RUNS = ["rtdetrx_dfire_s3407", "rtdetrx_pyrosdis_s3407"]
VERIFY_RUNS = ["rtdetrl_dfire_s3407", "rtdetrl_pyrosdis_s3407"]
TARGETS = {"dfire": ["pyro_sdis_caltest", "thesis_test"], "pyrosdis": ["d_fire_test_smokeonly", "thesis_test_smokeonly"]}
METHODS = ["M0_t_src", "D25_default", "M1c_kept_rate", "M1c_kept_rate_100", "M1a_quantile_pooled", "ORC_target_calval"]
OPJ = json.loads((HERE / "operating_points.json").read_text(encoding="utf-8"))
REFINE = json.loads((HERE / "label_free_threshold_refine.json").read_text(encoding="utf-8"))
PILOT = json.loads((HERE / "label_free_threshold_pilot.json").read_text(encoding="utf-8"))
CLEAN = json.loads((HERE / "label_free_clean_check.json").read_text(encoding="utf-8"))
W1_CUTOFF = CLEAN["part_B_shift_rule"]["null_cutoff_rule"]["cutoff_W1_imgmax"]

_gt, _det, _W = {}, {}, {}


def gt(tag):
    if tag not in _gt:
        _gt[tag] = op.load_gt(tag)
    return _gt[tag]


def dets(run, tag):
    k = (run, tag)
    if k not in _det:
        _det[k] = op.load_dets(pilot.DETS / f"{run}__{tag}.bbox.json")
    return _det[k]


def weights(tag, n_img):
    """Same draw as refine.main().weights: default_rng(SEED).multinomial once per target split."""
    if tag not in _W:
        rng = np.random.default_rng(SEED)
        _W[tag] = rng.multinomial(n_img, np.full(n_img, 1.0 / n_img), size=B).astype(float)
    assert _W[tag].shape == (B, n_img)
    return _W[tag]


def replay_draw(n_img):
    """The refine artefact's draw sequence (seed 0): 10 draws at n=25, 10 at n=50, then the first at n=100
    (copied from bn_adaptation_baseline.replay_draw; that module imports torch so it is not imported here)."""
    rng = np.random.default_rng(refine.SEED)
    for n in refine.SUBSAMPLE_N:
        if n == N_ADAPT:
            break
        for _ in range(refine.N_DRAWS):
            rng.choice(n_img, n, replace=False)
    return rng.choice(n_img, N_ADAPT, replace=False)


def subset_view(U, pick):
    """Unlabelled view restricted to the drawn images, exactly as the refine data-size curve builds it."""
    keep = np.zeros(U["n_img"], dtype=bool)
    keep[pick] = True
    m = keep[U["img"]]
    return {"score": U["score"][m], "img": U["img"][m], "rank": U["rank"][m], "n_dets_img": U["n_dets_img"], "n_img": int(len(pick))}


def m1c(view, stat_src):
    order = np.argsort(-view["score"], kind="stable")
    return refine.solve_threshold(view, order, "M1c_kept_rate", stat_src)


def m1a(view, q_pool):
    return float(np.quantile(view["score"], q_pool)) if len(view["score"]) else 1.0


def cell_names(runs):
    out = []
    for run in runs:
        src = "d_fire_calval" if "_dfire" in run else "pyro_sdis_calval"
        for tgt in TARGETS["dfire" if "_dfire" in run else "pyrosdis"]:
            cal = pilot.TARGET_CALVAL[tgt]
            files = {"source_calval": pilot.DETS / f"{run}__{src}.bbox.json", "target_calval": pilot.DETS / f"{run}__{cal}.bbox.json",
                     "target_test": pilot.DETS / f"{run}__{tgt}.bbox.json"}
            assert all(p.exists() for p in files.values()), files
            out.append((f"{run}__to__{tgt}", run, src, cal, tgt, {k: str(v) for k, v in files.items()}))
    return out


def run_cell(name, run, src, cal, tgt, files):
    t0 = time.time()
    s_ids, s_gt, s_cls = gt(src)
    u_ids, u_gt, _ = gt(cal)
    x_ids, x_gt, t_cls = gt(tgt)
    s_d, u_d, x_d = dets(run, src), dets(run, cal), dets(run, tgt)
    model_cls = {c for v in s_d.values() for c, _, _ in v} | {c for v in x_d.values() for c, _, _ in v}
    cls_s, cls_t = s_cls & model_cls, t_cls & model_cls
    # --- source calval (labelled) -> t_src, kappa, source pooled quantile ---------------------------
    S = refine.match_indexed(s_ids, s_gt, s_d, cls_s)
    Sc = op.pr_curve(S["score"], S["tp"], S["n_gt"])
    t_src = float(Sc[0][int(np.argmax(Sc[3]))])
    assert abs(t_src - OPJ["cells"][name]["source"]["t_f1opt"]) < 1e-9, (name, t_src)
    kappa = refine.source_statistic(refine.unlabelled_view(S), t_src, "M1c_kept_rate")
    q_pool = float(np.mean(S["score"] < t_src))
    # --- target calval: scores only for the rules; labels ONLY for ORC ------------------------------
    U = refine.match_indexed(u_ids, u_gt, u_d, cls_t)
    view = refine.unlabelled_view(U)
    pick = replay_draw(U["n_img"])
    thr = {"M0_t_src": t_src, "D25_default": 0.25, "M1c_kept_rate": m1c(view, kappa),
           "M1c_kept_rate_100": m1c(subset_view(U, pick), kappa), "M1a_quantile_pooled": m1a(view, q_pool)}
    Uc = op.pr_curve(U["score"], U["tp"], U["n_gt"])
    t_orc = float(Uc[0][int(np.argmax(Uc[3]))])
    thr["ORC_target_calval"] = t_orc
    # leakage: delete every target-calval annotation and recompute the label-free thresholds
    U0 = refine.match_indexed(u_ids, {}, u_d, cls_t)
    v0 = refine.unlabelled_view(U0)
    thr0 = {"M1c_kept_rate": m1c(v0, kappa), "M1c_kept_rate_100": m1c(subset_view(U0, pick), kappa), "M1a_quantile_pooled": m1a(v0, q_pool)}
    d_leak = max(abs(thr[k] - thr0[k]) for k in thr0)
    assert d_leak == 0.0, (name, d_leak)
    # --- shift diagnostic ----------------------------------------------------------------------------
    w1_imgmax = float(stats.wasserstein_distance(S["img_max"], U["img_max"]))
    w1_pooled = float(stats.wasserstein_distance(S["score"], U["score"])) if len(U["score"]) else None
    # --- target test (labelled evaluation) + bootstrap ----------------------------------------------
    X = refine.match_indexed(x_ids, x_gt, x_d, cls_t)
    W = weights(tgt, X["n_img"])
    Xc = op.pr_curve(X["score"], X["tp"], X["n_gt"])
    f1_topt = float(Xc[3].max())
    res, rep = {}, {}
    for m in METHODS:
        point, r = refine.evaluate(X, thr[m], W)
        assert abs(point["f1"] - op.at_threshold(*Xc, thr[m])["f1"]) < 1e-9, (name, m)
        res[m], rep[m] = {"threshold": thr[m], **point}, r
    f1_orc, f1_src = res["ORC_target_calval"]["f1"], res["M0_t_src"]["f1"]
    gap_point = f1_orc - f1_src
    gap_rep = rep["ORC_target_calval"]["f1"] - rep["M0_t_src"]["f1"]
    for m in METHODS:
        r = res[m]
        r["regret_vs_oracle"] = f1_orc - r["f1"]
        r["regret_vs_test_opt"] = f1_topt - r["f1"]
        r["abs_log10_ratio_to_oracle"] = abs(float(np.log10(max(thr[m], 1e-6) / max(t_orc, 1e-6))))
        r["ci95"] = {k: refine.ci(rep[m][k]) for k in ("f1", "sensitivity", "false_alarm_rate")}
        r["regret_vs_oracle_ci95"] = refine.ci(rep["ORC_target_calval"]["f1"] - rep[m]["f1"])
        if m != "M0_t_src":
            d = rep[m]["f1"] - rep["M0_t_src"]["f1"]
            r["diff_vs_t_src"] = {"f1": r["f1"] - f1_src, "ci95": refine.ci(d), "p_boot_le_0": float(np.mean(d <= 0))}
            valid = gap_rep > 0.005
            gc = np.where(valid, d / np.where(valid, gap_rep, 1.0), np.nan)
            r["gap_closed"] = {"point": (r["f1"] - f1_src) / gap_point if gap_point > 1e-9 else None, "ci95": refine.ci(gc),
                               "n_valid_replicates": int(valid.sum()), "unstable": bool(valid.sum() < 0.95 * B)}
    res["ORC_target_calval"]["gap_ci95"] = refine.ci(gap_rep)
    # --- LaECE_0 (identity map, kept detections, top-100 pre-filter, toolbox) -------------------------
    raw = [d for d in json.loads(Path(files["target_test"]).read_text(encoding="utf-8")) if d["category_id"] in cls_t]
    raw_f = refine.top_per_image(raw, refine.LAECE_TOP_PER_IMAGE)
    for m in METHODS:
        kept = [d for d in raw_f if d["score"] >= thr[m]]
        res[m]["LaECE_0"] = pilot._q(pilot.laece0, str(pilot.GT / f"{tgt}.json"), kept)
        res[m]["LaECE_0_n_kept"] = len(kept)
    cell = {"run": run, "family": "rtdetr" if run.startswith("rtdetr") else "yolo", "model": run.split("_")[0], "train_set": run.split("_")[1],
            "source_calval": src, "target_calval_unlabelled": cal, "target_test": tgt, "target": pilot.dataset_of(tgt), "files": files,
            "classes_evaluated": sorted(cls_t), "classes_source": sorted(cls_s),
            "n": {"source_dets": int(len(S["score"])), "source_gt": S["n_gt"], "source_images": S["n_img"],
                  "target_calval_dets": int(len(U["score"])), "target_calval_images": U["n_img"], "target_calval_dets_in_100_draw": int(len(subset_view(U, pick)["score"])),
                  "target_test_dets": int(len(X["score"])), "target_test_gt": X["n_gt"], "target_test_images": X["n_img"]},
            "source": {"t_src": t_src, "f1_at_t_src_on_source_calval": float(Sc[3].max()), "kappa_kept_dets_per_image_at_t_src": kappa,
                       "quantile_pooled_at_t_src": q_pool},
            "draw_100": {"seed": SEED, "rule": "default_rng(0); 10 draws n=25, 10 draws n=50, first draw n=100 (refine data-size curve, bn_adaptation_baseline.replay_draw)",
                         "image_indices_sorted": sorted(int(i) for i in pick)},
            "target_calval_labels_for_oracle_only": {"t_oracle": t_orc, "f1_at_oracle_on_calval": float(Uc[3].max())},
            "target_test_summary": {"t_test_optimal": float(Xc[0][int(np.argmax(Xc[3]))]), "f1_test_optimal": f1_topt,
                                    "gap_oracle_minus_t_src": gap_point, "gap_ci95": res["ORC_target_calval"]["gap_ci95"]},
            "shift_diagnostics": {"W1_imgmax": w1_imgmax, "W1_pooled": w1_pooled, "W1_cutoff_in_domain_null": W1_CUTOFF, "above_cutoff": bool(w1_imgmax > W1_CUTOFF),
                                  "abs_t_src_minus_t_oracle": abs(t_src - t_orc)},
            "leakage_check": {"max_abs_threshold_diff_after_deleting_target_calval_labels": d_leak, "passed": d_leak == 0.0},
            "laece": {"n_dets": len(raw), "n_after_prefilter": len(raw_f), "n_removed_by_prefilter": len(raw) - len(raw_f)},
            "methods": res, "seconds": round(time.time() - t0, 1)}
    print(f"{name}: t_src {t_src:.3f} kappa {kappa:.3f} | t M1c {thr['M1c_kept_rate']:.3f} M1c100 {thr['M1c_kept_rate_100']:.3f} M1a {thr['M1a_quantile_pooled']:.3f} ORC {t_orc:.3f} | "
          f"F1 src {f1_src:.3f} D25 {res['D25_default']['f1']:.3f} M1c {res['M1c_kept_rate']['f1']:.3f} M1c100 {res['M1c_kept_rate_100']['f1']:.3f} "
          f"M1a {res['M1a_quantile_pooled']['f1']:.3f} ORC {f1_orc:.3f} | W1 {w1_imgmax:.4f} | {time.time() - t0:.0f}s", flush=True)
    return cell


def verify(name, cell):
    """Assert the code path reproduces the refine / pilot artefacts for an existing cell."""
    R, P = REFINE["cells"][name], PILOT["cells"][name]
    m = cell["methods"]
    d = {"t_M1c": abs(m["M1c_kept_rate"]["threshold"] - R["methods"]["M1c_kept_rate"]["threshold"]),
         "t_ORC": abs(m["ORC_target_calval"]["threshold"] - R["methods"]["ORC_target_calval"]["threshold"]),
         "t_src": abs(cell["source"]["t_src"] - R["source"]["t_src"]),
         "kappa": abs(cell["source"]["kappa_kept_dets_per_image_at_t_src"] - R["source"]["statistics_at_t_src"]["M1c_kept_rate"]),
         "f1_max_over_t_src_D25_M1c_ORC": max(abs(m[k]["f1"] - R["methods"][k]["f1"]) for k in ("M0_t_src", "D25_default", "M1c_kept_rate", "ORC_target_calval")),
         "regret_M1c_ci95_max": max(abs(a - b) for a, b in zip(m["M1c_kept_rate"]["regret_vs_oracle_ci95"], R["methods"]["M1c_kept_rate"]["regret_vs_oracle_ci95"])),
         "gap_closed_M1c": (abs(m["M1c_kept_rate"]["gap_closed"]["point"] - R["methods"]["M1c_kept_rate"]["gap_closed"]["point"])
                            if m["M1c_kept_rate"]["gap_closed"]["point"] is not None and R["methods"]["M1c_kept_rate"]["gap_closed"]["point"] is not None
                            else (0.0 if m["M1c_kept_rate"]["gap_closed"]["point"] is None and R["methods"]["M1c_kept_rate"]["gap_closed"]["point"] is None else 1.0)),
         "LaECE_0_max_over_t_src_D25_M1c_ORC": max(abs(m[k]["LaECE_0"] - R["methods"][k]["LaECE_0"]) for k in ("M0_t_src", "D25_default", "M1c_kept_rate", "ORC_target_calval")),
         "t_M1c_100_vs_refine_draw0": abs(m["M1c_kept_rate_100"]["threshold"] - R["unlabelled_data_sensitivity"]["100"]["M1c_kept_rate"]["threshold_draws"][0]),
         "regret_M1c_100_vs_refine_draw0": abs(m["M1c_kept_rate_100"]["regret_vs_oracle"] - R["unlabelled_data_sensitivity"]["100"]["M1c_kept_rate"]["regret_draws"][0]),
         "t_M1a_vs_pilot": abs(m["M1a_quantile_pooled"]["threshold"] - P["methods"]["M1a_quantile_pooled"]["threshold"]),
         "f1_M1a_vs_pilot": abs(m["M1a_quantile_pooled"]["f1"] - P["methods"]["M1a_quantile_pooled"]["box"]["f1"]),
         "LaECE_0_M1a_vs_pilot": abs(m["M1a_quantile_pooled"]["LaECE_0"] - P["methods"]["M1a_quantile_pooled"]["LaECE_0"]),
         "W1_imgmax_vs_pilot": abs(cell["shift_diagnostics"]["W1_imgmax"] - P["shift_diagnostics"]["wasserstein_imgmax"]),
         "W1_pooled_vs_pilot": abs(cell["shift_diagnostics"]["W1_pooled"] - P["shift_diagnostics"]["wasserstein_pooled"])}
    tol = {"t_M1c": 1e-6, "t_ORC": 1e-6, "t_src": 1e-9, "kappa": 1e-9, "f1_max_over_t_src_D25_M1c_ORC": 1e-9, "regret_M1c_ci95_max": 1e-9, "gap_closed_M1c": 1e-9,
           "LaECE_0_max_over_t_src_D25_M1c_ORC": 1e-9, "t_M1c_100_vs_refine_draw0": 0.0, "regret_M1c_100_vs_refine_draw0": 1e-9, "t_M1a_vs_pilot": 1e-9,
           "f1_M1a_vs_pilot": 1e-9, "LaECE_0_M1a_vs_pilot": 1e-9, "W1_imgmax_vs_pilot": 1e-9, "W1_pooled_vs_pilot": 1e-9}
    ok = {k: bool(d[k] <= tol[k]) for k in d}
    assert all(ok.values()), (name, {k: d[k] for k in d if not ok[k]})
    return {"abs_diffs": d, "tolerances": tol, "passed": True}


def rtdetrl_reference(name_x):
    """The refine / pilot entries of the RT-DETR-l seed-3407 cell of the same direction."""
    name_l = name_x.replace("rtdetrx_", "rtdetrl_")
    R, P = REFINE["cells"][name_l], PILOT["cells"][name_l]
    m = R["methods"]
    ref = {"cell": name_l, "artefact": "label_free_threshold_refine.json cells.<cell> (+ label_free_threshold_pilot.json for M1a / W1)",
           "t_src": R["source"]["t_src"], "kappa": R["source"]["statistics_at_t_src"]["M1c_kept_rate"],
           "t_M1c": m["M1c_kept_rate"]["threshold"], "t_M1c_100_draw0": R["unlabelled_data_sensitivity"]["100"]["M1c_kept_rate"]["threshold_draws"][0],
           "t_M1a": P["methods"]["M1a_quantile_pooled"]["threshold"], "t_ORC": m["ORC_target_calval"]["threshold"],
           "f1": {"M0_t_src": m["M0_t_src"]["f1"], "D25_default": m["D25_default"]["f1"], "M1c_kept_rate": m["M1c_kept_rate"]["f1"],
                  "M1c_kept_rate_100": m["ORC_target_calval"]["f1"] - R["unlabelled_data_sensitivity"]["100"]["M1c_kept_rate"]["regret_draws"][0],
                  "M1a_quantile_pooled": P["methods"]["M1a_quantile_pooled"]["box"]["f1"], "ORC_target_calval": m["ORC_target_calval"]["f1"]},
           "regret": {"M0_t_src": m["M0_t_src"]["regret_vs_oracle"], "D25_default": m["D25_default"]["regret_vs_oracle"],
                      "M1c_kept_rate": m["M1c_kept_rate"]["regret_vs_oracle"], "M1c_kept_rate_100": R["unlabelled_data_sensitivity"]["100"]["M1c_kept_rate"]["regret_draws"][0],
                      "M1a_quantile_pooled": P["methods"]["M1a_quantile_pooled"]["regret_vs_oracle"]},
           "regret_ci95": {k: m[k]["regret_vs_oracle_ci95"] for k in ("M0_t_src", "D25_default", "M1c_kept_rate")},
           "gap_closed_M1c": m["M1c_kept_rate"]["gap_closed"]["point"], "gap_closed_M1c_ci95": m["M1c_kept_rate"]["gap_closed"]["ci95"],
           "LaECE_0": {k: m[k]["LaECE_0"] for k in ("M0_t_src", "D25_default", "M1c_kept_rate", "ORC_target_calval")},
           "W1_imgmax": P["shift_diagnostics"]["wasserstein_imgmax"], "W1_pooled": P["shift_diagnostics"]["wasserstein_pooled"],
           "regret_100_mean_over_10_draws": R["unlabelled_data_sensitivity"]["100"]["M1c_kept_rate"]["regret_mean"]}
    return ref


def main():
    t_start = time.time()
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "protocol": __doc__, "iou": op.IOU,
           "bootstrap": {"B": B, "seed": SEED, "unit": "target-test image (multinomial weights, identical draw to the refine artefact)", "ci": "percentile 2.5-97.5"},
           "laece_prefilter_top_per_image": refine.LAECE_TOP_PER_IMAGE, "W1_cutoff_in_domain_null": W1_CUTOFF,
           "methods": METHODS, "verification": {}, "cells": {}, "comparison_with_rtdetrl": {}}
    for name, run, src, cal, tgt, files in cell_names(VERIFY_RUNS):
        cell = run_cell(name, run, src, cal, tgt, files)
        out["verification"][name] = verify(name, cell)
        print(f"  verified against refine/pilot: max|d| = {max(out['verification'][name]['abs_diffs'].values()):.2e}", flush=True)
    for name, run, src, cal, tgt, files in cell_names(NEW_RUNS):
        cell = run_cell(name, run, src, cal, tgt, files)
        out["cells"][name] = cell
        ref = rtdetrl_reference(name)
        m = cell["methods"]
        out["comparison_with_rtdetrl"][name] = {
            "rtdetrl_cell": ref["cell"], "rtdetrl": ref,
            "diff_x_minus_l": {"t_src": cell["source"]["t_src"] - ref["t_src"], "kappa": cell["source"]["kappa_kept_dets_per_image_at_t_src"] - ref["kappa"],
                               "t_M1c": m["M1c_kept_rate"]["threshold"] - ref["t_M1c"], "t_ORC": m["ORC_target_calval"]["threshold"] - ref["t_ORC"],
                               "f1": {k: m[k]["f1"] - ref["f1"][k] for k in METHODS},
                               "regret": {k: m[k]["regret_vs_oracle"] - ref["regret"][k] for k in ref["regret"]},
                               "gap_closed_M1c": (m["M1c_kept_rate"]["gap_closed"]["point"] - ref["gap_closed_M1c"]) if (m["M1c_kept_rate"]["gap_closed"]["point"] is not None and ref["gap_closed_M1c"] is not None) else None,
                               "W1_imgmax": cell["shift_diagnostics"]["W1_imgmax"] - ref["W1_imgmax"]}}
    out["verification_summary"] = {"n_cells": len(out["verification"]), "passed": all(v["passed"] for v in out["verification"].values()),
                                   "max_abs_diff_t_M1c": max(v["abs_diffs"]["t_M1c"] for v in out["verification"].values()),
                                   "max_abs_diff_f1": max(v["abs_diffs"]["f1_max_over_t_src_D25_M1c_ORC"] for v in out["verification"].values()),
                                   "max_abs_diff_any": max(max(v["abs_diffs"].values()) for v in out["verification"].values())}
    out["leakage_check"] = {"cells": len(out["cells"]), "passed": all(c["leakage_check"]["passed"] for c in out["cells"].values())}
    out["seconds_total"] = round(time.time() - t_start, 1)
    (HERE / "label_free_rtdetrx.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    f3 = lambda x: "" if x is None else f"{x:.3f}"
    cis = lambda c: "" if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]"
    L = ["# Label-free threshold transfer - RT-DETR-x cells (machine-written by label_free_rtdetrx.py)", "",
         f"Created {out['created']}. Refine protocol reused unchanged; verification on the four RT-DETR-l seed-3407 cells of the same directions passed "
         f"(max |dt_M1c| = {out['verification_summary']['max_abs_diff_t_M1c']:.2e}, max |dF1| = {out['verification_summary']['max_abs_diff_f1']:.2e}, "
         f"max |d| over every checked quantity = {out['verification_summary']['max_abs_diff_any']:.2e}). Leakage check passed: {out['leakage_check']['passed']}. "
         f"Bootstrap B = {B}, seed {SEED}. W1 cut-off (in-domain null, label_free_clean_check.json) = {W1_CUTOFF:.4f}.", "",
         "## Thresholds", "", "| cell | t_src | kappa | t M1c full | t M1c 100 | t M1a | t ORC | W1 imgmax (above cut-off) | |t_src - t_ORC| |", "|---|---|---|---|---|---|---|---|---|"]
    for c, e in out["cells"].items():
        m, s = e["methods"], e["shift_diagnostics"]
        L.append(f"| {c} | {f3(e['source']['t_src'])} | {f3(e['source']['kappa_kept_dets_per_image_at_t_src'])} | {f3(m['M1c_kept_rate']['threshold'])} | {f3(m['M1c_kept_rate_100']['threshold'])} | "
                 f"{f3(m['M1a_quantile_pooled']['threshold'])} | {f3(m['ORC_target_calval']['threshold'])} | {s['W1_imgmax']:.4f} ({'yes' if s['above_cutoff'] else 'no'}) | {f3(s['abs_t_src_minus_t_oracle'])} |")
    L += ["", "## F1 on the target test (box, IoU 0.5) with 95 % image-bootstrap CI; regret = F1(ORC) - F1(method)", "",
          "| cell | method | threshold | F1 [CI] | regret [CI] | gain vs t_src [CI] | gap closed [CI] | sens | FAR | LaECE_0 (n kept) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c, e in out["cells"].items():
        for k in METHODS:
            r = e["methods"][k]
            gc = r.get("gap_closed")
            L.append(f"| {c} | {k} | {f3(r['threshold'])} | {f3(r['f1'])} {cis(r['ci95']['f1'])} | {f3(r['regret_vs_oracle'])} {cis(r['regret_vs_oracle_ci95'])} | "
                     f"{f3(r['diff_vs_t_src']['f1']) if 'diff_vs_t_src' in r else ''} {cis(r['diff_vs_t_src']['ci95']) if 'diff_vs_t_src' in r else ''} | "
                     f"{(f3(gc['point']) + ' ' + cis(gc['ci95']) + (' (unstable)' if gc['unstable'] else '')) if gc else ''} | {f3(r['sensitivity'])} | {f3(r['false_alarm_rate'])} | "
                     f"{f3(r['LaECE_0'])} ({r['LaECE_0_n_kept']}) |")
    L += ["", "## RT-DETR-x vs RT-DETR-l (same direction, seed 3407; l values from the refine / pilot artefacts)", "",
          "| direction | model | t_src | kappa | t M1c | t ORC | F1 t_src | F1 D25 | F1 M1c | F1 M1c100 | F1 M1a | F1 ORC | regret M1c [CI] | gap closed | W1 imgmax |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, e in out["cells"].items():
        m, ref = e["methods"], out["comparison_with_rtdetrl"][c]["rtdetrl"]
        L.append(f"| {c.split('__to__')[0].split('_', 1)[1]} -> {e['target_test']} | rtdetrx | {f3(e['source']['t_src'])} | {f3(e['source']['kappa_kept_dets_per_image_at_t_src'])} | {f3(m['M1c_kept_rate']['threshold'])} | {f3(m['ORC_target_calval']['threshold'])} | "
                 f"{f3(m['M0_t_src']['f1'])} | {f3(m['D25_default']['f1'])} | {f3(m['M1c_kept_rate']['f1'])} | {f3(m['M1c_kept_rate_100']['f1'])} | {f3(m['M1a_quantile_pooled']['f1'])} | {f3(m['ORC_target_calval']['f1'])} | "
                 f"{f3(m['M1c_kept_rate']['regret_vs_oracle'])} {cis(m['M1c_kept_rate']['regret_vs_oracle_ci95'])} | {f3(m['M1c_kept_rate']['gap_closed']['point'])} | {e['shift_diagnostics']['W1_imgmax']:.4f} |")
        L.append(f"| | rtdetrl | {f3(ref['t_src'])} | {f3(ref['kappa'])} | {f3(ref['t_M1c'])} | {f3(ref['t_ORC'])} | {f3(ref['f1']['M0_t_src'])} | {f3(ref['f1']['D25_default'])} | {f3(ref['f1']['M1c_kept_rate'])} | "
                 f"{f3(ref['f1']['M1c_kept_rate_100'])} | {f3(ref['f1']['M1a_quantile_pooled'])} | {f3(ref['f1']['ORC_target_calval'])} | {f3(ref['regret']['M1c_kept_rate'])} {cis(ref['regret_ci95']['M1c_kept_rate'])} | {f3(ref['gap_closed_M1c'])} | {ref['W1_imgmax']:.4f} |")
    L += ["", "## Verification cells (RT-DETR-l seed 3407): max |difference| vs refine / pilot artefacts", "", "| cell | quantity | abs diff | tolerance |", "|---|---|---|---|"]
    for c, v in out["verification"].items():
        for k, d in v["abs_diffs"].items():
            L.append(f"| {c} | {k} | {d:.2e} | {v['tolerances'][k]:.0e} |")
    (HERE / "label_free_rtdetrx.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:30]))
    print("artefact ->", HERE / "label_free_rtdetrx.json", "and .md", f"({out['seconds_total']} s)")


if __name__ == "__main__":
    main()
