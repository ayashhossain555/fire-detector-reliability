"""Label-free threshold transfer — clean-split robustness, shift-diagnostic decision rule, Table-4 tallies
(2026-09-13, CPU only).  Reuses loaders, matching (op.match at IoU 0.5 via refine.match_indexed), the kept-rate
rule (refine.solve_threshold, variant M1c_kept_rate), the oracle and the bootstrap machinery of
label_free_threshold_refine.py / label_free_threshold_pilot.py UNCHANGED (read-only imports).

PART A — clean-split robustness of the kept-rate rule on the D-Fire target.
  Referee objection: d_fire_calval (drawn from the official D-Fire training split) holds near-duplicates of the
  official test images, so both the labelled oracle and the rule's unlabelled pool see test-adjacent frames.
  Re-evaluation for every Pyro-SDIS-trained run whose target is D-Fire smoke:
    unlabelled pool AND oracle-fitting set = the de-duplicated calibration split (image list of
        coco_gt/d_fire_calval_dedup.json, a subset of d_fire_calval with identical image ids and file names —
        asserted); detections = detections/<run>__d_fire_calval.bbox.json filtered to those image ids;
    test = the clean smoke-only test split (detections/<run>__d_fire_test_smokeonly_dedup.bbox.json +
        coco_gt/d_fire_test_smokeonly_dedup.json).
  The OFFICIAL-split numbers (pool = all 1722 d_fire_calval images, test = d_fire_test_smokeonly) are recomputed
  here with the refine protocol for every cell and, for every cell present in label_free_threshold_refine.json,
  asserted equal to it (thresholds 1e-6, F1 1e-9) before the clean run is trusted.  v8m_pyrosdis_s3407 and
  y11m_pyrosdis_s3407 were exported after the refine/pilot artefacts (44 cells = 22 runs) and have no refine cell;
  their official-split numbers exist only as computed here (flagged refine_cell_present = false).
  Reported per cell and as means: F1 at t_src / kept-rate / oracle (+ D25 for reference), regret, gap closed,
  the official-split numbers of the same cells from the refine artefact, image-level bootstrap CIs of the clean
  test (B = 1000, seed 0, multinomial image weights shared across cells) and a cell-level bootstrap of the mean
  regret.  Extra decomposition: the OFFICIAL-split thresholds (t_M1c, t_ORC of the refine artefact) evaluated on
  the clean test, so the effect of changing the unlabelled pool is separated from the effect of changing the test.

PART B — a decision rule for the shift diagnostic (1-D Wasserstein distance W1 between the per-image maximum
  score on the source calval and on the target calval; per-image max = op.match per_img max over the evaluated
  classes, exactly as the pilot; recomputed here and asserted equal to the pilot's wasserstein_imgmax).
  In-domain null: (i) for each of the 24 runs, W1 between the per-image max on its own source calval and on its
  own source test split (the in-domain cell of operating_points.json; D-Fire-trained runs: d_fire_test, with
  d_fire_test_dedup recorded as a secondary value; dfiredd runs: d_fire_calval_dedup vs d_fire_test_dedup);
  (ii) for every Pyro-SDIS-trained run, W1 between the calibration cameras and the held-out cameras of each of
  the four camera-disjoint folds of camera_conformal.json (pooled pyro_sdis calval + caltest, cameras from
  datasets/pyro_sdis_image_meta.json).  Cut-off = max in-domain W1 (no cross-domain cell is used to fit it).
  Reported: null distribution (max, p95), cells above/below the cut-off with mean regret(t_src) and mean |drift|,
  ROC AUC of W1 for |t_src - t_oracle| > 0.1 and for regret(t_src) > 0.02, and — because a FITTED cut-off is
  involved there — a leave-one-target-out evaluation of the cut-off that maximises balanced accuracy on the
  training targets (targets = d_fire / pyro_sdis / thesis).

PART C — Table-4 tallies: for every rule of label_free_threshold_refine.json summary.groups.all and the pilot-only
  rules (quantile pooled / image-max, every EM mixture variant), n_cells, beats t_src, worse than t_src and
  ties = n - beats - worse (recounted from the per-cell F1 values with the same 1e-12 tolerance, asserted equal
  to the stored tallies).

Writes analysis/label_free_clean_check.json (+ .md).  Does not modify any existing file.
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

B, SEED = 1000, 0
GT, DETS = pilot.GT, pilot.DETS
REFINE = json.loads((HERE / "label_free_threshold_refine.json").read_text(encoding="utf-8"))
PILOT = json.loads((HERE / "label_free_threshold_pilot.json").read_text(encoding="utf-8"))
OPJ = json.loads((HERE / "operating_points.json").read_text(encoding="utf-8"))
CAMJ = json.loads((HERE / "camera_conformal.json").read_text(encoding="utf-8"))
META = json.loads((HERE / "datasets" / "pyro_sdis_image_meta.json").read_text(encoding="utf-8"))

CORE_9 = ["v8n_pyrosdis_s3407", "v8s_pyrosdis_s3407", "v8m_pyrosdis_s3407", "y11n_pyrosdis_s3407", "y11s_pyrosdis_s3407",
          "v8s_pyrosdis_s1337", "y11s_pyrosdis_s1337", "rtdetrl_pyrosdis_s1337", "rtdetrl_pyrosdis_s3407"]
EXTRA = ["rtdetrl_pyrosdis_s3407_p60", "y11m_pyrosdis_s3407"]
DRIFT_CUT, REGRET_CUT = 0.1, 0.02

_gt, _det = {}, {}


def gt(tag):
    if tag not in _gt:
        _gt[tag] = op.load_gt(tag)
    return _gt[tag]


def dets(run, tag):
    k = (run, tag)
    if k not in _det:
        _det[k] = op.load_dets(DETS / f"{run}__{tag}.bbox.json")
    return _det[k]


def model_classes(run, tags):
    return {c for t in tags for v in dets(run, t).values() for c, _, _ in v}


def f1opt(M):
    c = op.pr_curve(M["score"], M["tp"], M["n_gt"])
    return float(c[0][int(np.argmax(c[3]))]), float(c[3].max())


def kept_rate_threshold(S, t_src, U):
    """M1c exactly as the refine script: source statistic at t_src, nearest attainable on the unlabelled target."""
    stat = refine.source_statistic(refine.unlabelled_view(S), t_src, "M1c_kept_rate")
    view = refine.unlabelled_view(U)
    order = np.argsort(-view["score"], kind="stable")
    return refine.solve_threshold(view, order, "M1c_kept_rate", stat), stat


def auc(score, label):
    """Mann-Whitney ROC AUC (ties count 1/2). None if one class is empty."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    p, n = s[y], s[~y]
    if len(p) == 0 or len(n) == 0:
        return None
    return float((np.mean(p[:, None] > n[None, :]) + 0.5 * np.mean(p[:, None] == n[None, :])))


# =============================================================================== PART A
def part_a():
    t0 = time.time()
    cal_ids, cal_gt, cal_cls = gt("d_fire_calval")
    dd_ids, dd_gt, dd_cls = gt("d_fire_calval_dedup")
    cal_j = json.loads((GT / "d_fire_calval.json").read_text(encoding="utf-8"))
    dd_j = json.loads((GT / "d_fire_calval_dedup.json").read_text(encoding="utf-8"))
    fn_cal = {im["id"]: im["file_name"] for im in cal_j["images"]}
    fn_dd = {im["id"]: im["file_name"] for im in dd_j["images"]}
    assert set(dd_ids) <= set(cal_ids), "dedup ids are not a subset of d_fire_calval ids"
    assert all(fn_cal[i] == fn_dd[i] for i in dd_ids), "id -> file_name differs between d_fire_calval and d_fire_calval_dedup"
    assert len(set(fn_dd.values())) == len(dd_ids) == 927, len(dd_ids)
    ann_cal = {(a["image_id"], a["category_id"], tuple(a["bbox"])) for a in cal_j["annotations"] if a["image_id"] in set(dd_ids)}
    ann_dd = {(a["image_id"], a["category_id"], tuple(a["bbox"])) for a in dd_j["annotations"]}
    mapping_check = {"n_dedup_images": len(dd_ids), "n_calval_images": len(cal_ids), "ids_subset": True, "same_file_name_for_same_id": True,
                     "annotations_identical_on_dedup_images": ann_cal == ann_dd, "n_annotations_dedup": len(ann_dd),
                     "n_annotations_calval_restricted_to_dedup_ids": len(ann_cal)}
    x_ids, x_gt, x_cls = gt("d_fire_test_smokeonly_dedup")
    assert x_cls == {2}
    dd_set = set(dd_ids)
    runs = [r for r in CORE_9 + EXTRA if (DETS / f"{r}__d_fire_test_smokeonly_dedup.bbox.json").exists() and (DETS / f"{r}__d_fire_calval.bbox.json").exists()]
    not_evaluable = [{"run": r, "why": "missing detections"} for r in CORE_9 + EXTRA if r not in runs]
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(len(x_ids), np.full(len(x_ids), 1.0 / len(x_ids)), size=B).astype(float)
    cells, reps, repro = {}, {}, {}
    xo_ids, xo_gt, xo_cls = gt("d_fire_test_smokeonly")
    assert xo_cls == {2}
    for run in runs:
        name = f"{run}__to__d_fire_test_smokeonly"
        R = REFINE["cells"].get(name)
        s_ids, s_gt, s_cls = gt("pyro_sdis_calval")
        mcls = model_classes(run, ["pyro_sdis_calval", "d_fire_test_smokeonly_dedup"])
        cls_s, cls_t = s_cls & mcls, x_cls & mcls
        assert cls_t == {2}
        S = refine.match_indexed(s_ids, s_gt, dets(run, "pyro_sdis_calval"), cls_s)
        t_src, f1_src_cal = f1opt(S)
        t_src_ref = OPJ["cells"][f"{run}__to__pyro_sdis_caltest"]["source"]["t_f1opt"]
        assert abs(t_src - t_src_ref) < 1e-9, (run, t_src, t_src_ref)
        # --- OFFICIAL split recomputed with the refine protocol (pool = all 1722 calval images, test = d_fire_test_smokeonly)
        U_off = refine.match_indexed(cal_ids, cal_gt, dets(run, "d_fire_calval"), cls_t)
        t_m1c_off, stat_src = kept_rate_threshold(S, t_src, U_off)
        t_orc_off, _ = f1opt(U_off)
        Xo = refine.match_indexed(xo_ids, xo_gt, dets(run, "d_fire_test_smokeonly"), cls_t)
        Xoc = op.pr_curve(Xo["score"], Xo["tp"], Xo["n_gt"])
        f1o = {m: op.at_threshold(*Xoc, t)["f1"] for m, t in (("M0_t_src", t_src), ("D25_default", 0.25), ("M1c_kept_rate", t_m1c_off), ("ORC_target_calval", t_orc_off))}
        official = {"refine_cell_present": R is not None, "t_src": t_src, "t_M1c": t_m1c_off, "t_ORC": t_orc_off,
                    "f1_t_src": f1o["M0_t_src"], "f1_M1c": f1o["M1c_kept_rate"], "f1_ORC": f1o["ORC_target_calval"], "f1_D25": f1o["D25_default"],
                    "regret_t_src": f1o["ORC_target_calval"] - f1o["M0_t_src"], "regret_M1c": f1o["ORC_target_calval"] - f1o["M1c_kept_rate"],
                    "regret_D25": f1o["ORC_target_calval"] - f1o["D25_default"],
                    "gap_closed_M1c": (f1o["M1c_kept_rate"] - f1o["M0_t_src"]) / (f1o["ORC_target_calval"] - f1o["M0_t_src"]) if f1o["ORC_target_calval"] - f1o["M0_t_src"] > 1e-9 else None,
                    "n_target_calval_images": U_off["n_img"], "n_target_test_images": Xo["n_img"], "n_target_test_gt": Xo["n_gt"],
                    "source": "recomputed here with the refine protocol; asserted equal to label_free_threshold_refine.json cells.<cell>.methods.* where that cell exists"}
        if R is not None:
            assert sorted(cls_t) == R["classes_evaluated"] and abs(t_src - R["source"]["t_src"]) < 1e-9
            d_m1c = abs(t_m1c_off - R["methods"]["M1c_kept_rate"]["threshold"])
            d_orc = abs(t_orc_off - R["methods"]["ORC_target_calval"]["threshold"])
            assert d_m1c < 1e-6, (run, t_m1c_off, R["methods"]["M1c_kept_rate"]["threshold"])
            assert d_orc < 1e-6, (run, t_orc_off, R["methods"]["ORC_target_calval"]["threshold"])
            d_f1 = max(abs(f1o[m] - R["methods"][m]["f1"]) for m in f1o)
            assert d_f1 < 1e-9, (run, d_f1)
            repro[name] = {"t_M1c_here": t_m1c_off, "t_M1c_refine": R["methods"]["M1c_kept_rate"]["threshold"], "abs_diff": d_m1c,
                           "t_ORC_here": t_orc_off, "t_ORC_refine": R["methods"]["ORC_target_calval"]["threshold"], "abs_diff_orc": d_orc, "max_abs_diff_f1": d_f1, "passed": True}
            official["regret_M1c_ci95_refine"] = R["methods"]["M1c_kept_rate"]["regret_vs_oracle_ci95"]
            official["gap_closed_M1c_ci95_refine"] = R["methods"]["M1c_kept_rate"]["gap_closed"]["ci95"]
            official["f1_M1c_ci95_refine"] = R["methods"]["M1c_kept_rate"]["ci95"]["f1"]
        # --- clean: dedup calval as the unlabelled pool and the oracle set --------------------------------------
        d_all = dets(run, "d_fire_calval")
        assert set(d_all) <= set(cal_ids)
        d_dd = {i: v for i, v in d_all.items() if i in dd_set}
        U = refine.match_indexed(dd_ids, dd_gt, d_dd, cls_t)
        assert U["n_img"] == 927
        t_m1c, stat_chk = kept_rate_threshold(S, t_src, U)
        assert stat_chk == stat_src
        U0 = refine.match_indexed(dd_ids, {}, d_dd, cls_t)          # leakage: labels deleted -> identical threshold
        t_m1c_0, _ = kept_rate_threshold(S, t_src, U0)
        assert t_m1c_0 == t_m1c
        t_orc, f1_orc_cal = f1opt(U)
        X = refine.match_indexed(x_ids, x_gt, dets(run, "d_fire_test_smokeonly_dedup"), cls_t)
        assert X["n_img"] == 1991
        Xc = op.pr_curve(X["score"], X["tp"], X["n_gt"])
        thr = {"M0_t_src": t_src, "D25_default": 0.25, "M1c_kept_rate": t_m1c, "ORC_target_calval": t_orc,
               "M1c_official_threshold_on_clean_test": t_m1c_off, "ORC_official_threshold_on_clean_test": t_orc_off}
        res, rep = {}, {}
        for m, t in thr.items():
            point, r = refine.evaluate(X, t, W)
            assert abs(point["f1"] - op.at_threshold(*Xc, t)["f1"]) < 1e-9
            res[m], rep[m] = {"threshold": t, **point}, r
        f1_src, f1_orc = res["M0_t_src"]["f1"], res["ORC_target_calval"]["f1"]
        gap = f1_orc - f1_src
        gap_rep = rep["ORC_target_calval"]["f1"] - rep["M0_t_src"]["f1"]
        for m in thr:
            r = res[m]
            r["regret_vs_oracle"] = f1_orc - r["f1"]
            r["regret_vs_oracle_ci95"] = refine.ci(rep["ORC_target_calval"]["f1"] - rep[m]["f1"])
            r["f1_ci95"] = refine.ci(rep[m]["f1"])
            rep[m]["regret"] = rep["ORC_target_calval"]["f1"] - rep[m]["f1"]
            if m != "M0_t_src":
                d = rep[m]["f1"] - rep["M0_t_src"]["f1"]
                r["diff_vs_t_src"] = {"f1": r["f1"] - f1_src, "ci95": refine.ci(d), "p_boot_le_0": float(np.mean(d <= 0))}
                valid = gap_rep > 0.005
                gc = np.where(valid, d / np.where(valid, gap_rep, 1.0), np.nan)
                r["gap_closed"] = {"point": (r["f1"] - f1_src) / gap if gap > 1e-9 else None, "ci95": refine.ci(gc),
                                   "n_valid_replicates": int(valid.sum()), "unstable": bool(valid.sum() < 0.95 * B)}
                rep[m]["diff_vs_t_src"], rep[m]["gap_closed"] = d, gc
        cells[name] = {"run": run, "family": "rtdetr" if run.startswith("rtdetr") else "yolo", "model": run.split("_")[0], "in_core_9": run in CORE_9,
                       "refine_cell_present": R is not None,
                       "source_calval": "pyro_sdis_calval", "unlabelled_pool_and_oracle_set": "d_fire_calval_dedup (927 images; detections filtered from <run>__d_fire_calval)",
                       "target_test": "d_fire_test_smokeonly_dedup", "classes_evaluated": sorted(cls_t),
                       "n": {"source_images": S["n_img"], "source_dets": int(len(S["score"])), "pool_images": U["n_img"], "pool_dets": int(len(U["score"])),
                             "pool_gt": U["n_gt"], "test_images": X["n_img"], "test_dets": int(len(X["score"])), "test_gt": X["n_gt"]},
                       "source": {"t_src": t_src, "f1_at_t_src_on_source": f1_src_cal, "kept_dets_per_image_at_t_src": stat_src},
                       "oracle_on_clean_calval": {"t_oracle": t_orc, "f1_at_oracle_on_calval": f1_orc_cal},
                       "clean_test_summary": {"t_test_optimal": float(Xc[0][int(np.argmax(Xc[3]))]), "f1_test_optimal": float(Xc[3].max()),
                                              "gap_oracle_minus_t_src": gap, "gap_ci95": refine.ci(gap_rep)},
                       "leakage_check_passed": True, "methods": res, "official_split": official}
        reps[name] = rep
        print(f"A {name}: t_src {t_src:.3f} M1c off {t_m1c_off:.3f} clean {t_m1c:.3f} | ORC off {t_orc_off:.3f} clean {t_orc:.3f} | "
              f"F1 src {f1_src:.3f} M1c {res['M1c_kept_rate']['f1']:.3f} ORC {f1_orc:.3f} | official F1 src {official['f1_t_src']:.3f} M1c {official['f1_M1c']:.3f} ORC {official['f1_ORC']:.3f}", flush=True)

    def group(sel):
        g = {"n_cells": len(sel), "cells": sel}
        rng_c = np.random.default_rng(SEED)
        for m in ("M0_t_src", "D25_default", "M1c_kept_rate", "ORC_target_calval", "M1c_official_threshold_on_clean_test", "ORC_official_threshold_on_clean_test"):
            reg = np.array([cells[c]["methods"][m]["regret_vs_oracle"] for c in sel])
            f1 = np.array([cells[c]["methods"][m]["f1"] for c in sel])
            f1s = np.array([cells[c]["methods"]["M0_t_src"]["f1"] for c in sel])
            reg_rep = np.mean([reps[c][m]["regret"] for c in sel], axis=0)
            e = {"f1_mean": float(f1.mean()), "threshold_mean": float(np.mean([cells[c]["methods"][m]["threshold"] for c in sel])),
                 "regret_mean": float(reg.mean()), "regret_median": float(np.median(reg)), "regret_max": float(reg.max()),
                 "regret_sd_across_cells": float(reg.std(ddof=1)) if len(sel) > 1 else None,
                 "regret_mean_ci95_image_bootstrap": refine.ci(reg_rep),
                 "regret_mean_ci95_cell_bootstrap": refine.ci([reg[rng_c.integers(0, len(sel), len(sel))].mean() for _ in range(B)]) if len(sel) > 1 else None,
                 "n_beats_t_src": int(np.sum(f1 > f1s + 1e-12)), "n_worse_than_t_src": int(np.sum(f1 < f1s - 1e-12))}
            if m != "M0_t_src":
                d_rep = np.mean([reps[c][m]["diff_vs_t_src"] for c in sel], axis=0)
                gcs = np.array([v for v in (cells[c]["methods"][m]["gap_closed"]["point"] for c in sel) if v is not None])
                e["f1_gain_vs_t_src_mean"] = float((f1 - f1s).mean()); e["f1_gain_vs_t_src_ci95_image_bootstrap"] = refine.ci(d_rep)
                e["gap_closed_mean"] = float(gcs.mean()) if len(gcs) else None; e["gap_closed_median"] = float(np.median(gcs)) if len(gcs) else None
                e["gap_closed_mean_ci95_image_bootstrap"] = refine.ci(np.nanmean([reps[c][m]["gap_closed"] for c in sel], axis=0))
            g[m] = e
        off = {}
        for k in ("f1_t_src", "f1_M1c", "f1_ORC", "f1_D25", "regret_t_src", "regret_M1c", "regret_D25", "gap_closed_M1c", "t_src", "t_M1c", "t_ORC"):
            v = np.array([cells[c]["official_split"][k] for c in sel if cells[c]["official_split"][k] is not None], dtype=float)
            off[f"{k}_mean"] = float(v.mean()) if len(v) else None
        off["regret_M1c_sd_across_cells"] = float(np.std([cells[c]["official_split"]["regret_M1c"] for c in sel], ddof=1)) if len(sel) > 1 else None
        off["n_cells_with_refine_cell"] = int(sum(cells[c]["refine_cell_present"] for c in sel))
        off["source"] = "means over the same cells of the official-split numbers recomputed here (equal to the refine artefact where its cell exists)"
        g["official_split_means"] = off
        g["clean_minus_official"] = {"regret_M1c": g["M1c_kept_rate"]["regret_mean"] - off["regret_M1c_mean"],
                                     "regret_t_src": g["M0_t_src"]["regret_mean"] - off["regret_t_src_mean"],
                                     "gap_closed_M1c": (g["M1c_kept_rate"]["gap_closed_mean"] - off["gap_closed_M1c_mean"]) if off["gap_closed_M1c_mean"] is not None else None,
                                     "f1_ORC": g["ORC_target_calval"]["f1_mean"] - off["f1_ORC_mean"]}
        return g

    names = list(cells)
    groups = {"all": group(names), "core_9": group([c for c in names if cells[c]["in_core_9"]]), "in_refine_artefact": group([c for c in names if cells[c]["refine_cell_present"]]),
              "yolo": group([c for c in names if cells[c]["family"] == "yolo"]), "rtdetr": group([c for c in names if cells[c]["family"] == "rtdetr"])}
    groups = {k: v for k, v in groups.items() if v["n_cells"] > 0}
    return {"mapping_check": mapping_check, "reproduction_official_split": {"cells": repro, "n_cells": len(repro), "max_abs_diff_M1c": max(v["abs_diff"] for v in repro.values()),
                                                                             "max_abs_diff_ORC": max(v["abs_diff_orc"] for v in repro.values()), "max_abs_diff_f1": max(v["max_abs_diff_f1"] for v in repro.values()),
                                                                             "tolerance_threshold": 1e-6, "tolerance_f1": 1e-9, "passed": True,
                                                                             "cells_without_refine_counterpart": [c for c in cells if not cells[c]["refine_cell_present"]]},
            "not_evaluable": not_evaluable, "bootstrap": {"B": B, "seed": SEED, "unit": "clean target-test image (multinomial weights shared across cells)"},
            "cells": cells, "groups": groups, "seconds": round(time.time() - t0, 1)}


# =============================================================================== PART B
def img_max(run, tag, classes, ids=None, boxes=None):
    i, b, _ = gt(tag) if ids is None else (ids, boxes, None)
    return refine.match_indexed(i, b, dets(run, tag), classes)["img_max"]


def part_b():
    t0 = time.time()
    cross = {}
    max_w_diff = 0.0
    for name, R in REFINE["cells"].items():
        run, src, cal, tgt = R["run"], R["source_calval"], R["target_calval_unlabelled"], R["target_test"]
        cls_s, cls_t = set(R["classes_source"]), set(R["classes_evaluated"])
        s_max, u_max = img_max(run, src, cls_s), img_max(run, cal, cls_t)
        w = float(stats.wasserstein_distance(s_max, u_max))
        wp = PILOT["cells"][name]["shift_diagnostics"]["wasserstein_imgmax"]
        max_w_diff = max(max_w_diff, abs(w - wp))
        assert abs(w - wp) < 1e-9, (name, w, wp)
        t_src, t_orc = R["source"]["t_src"], R["target_calval_labels_for_oracle_only"]["t_oracle"]
        cross[name] = {"run": run, "family": R["family"], "target_dataset": pilot.dataset_of(tgt), "target_test": tgt,
                       "W1_imgmax": w, "W1_pooled": PILOT["cells"][name]["shift_diagnostics"]["wasserstein_pooled"],
                       "abs_drift_t_src_minus_t_oracle": abs(t_src - t_orc), "regret_t_src": R["methods"]["M0_t_src"]["regret_vs_oracle"],
                       "regret_M1c": R["methods"]["M1c_kept_rate"]["regret_vs_oracle"]}
    # --- in-domain null (i): own calval vs own test ---------------------------------------------------------------
    runs = sorted({R["run"] for R in REFINE["cells"].values()} | {c["run"] for c in OPJ["cells"].values()})
    assert len(runs) == 24, runs
    null_split, null_split_secondary = {}, {}
    for run in runs:
        if "pyrosdis" in run:
            src, tests = "pyro_sdis_calval", ["pyro_sdis_caltest"]
        elif "dfiredd" in run:
            src, tests = "d_fire_calval_dedup", ["d_fire_test_dedup", "d_fire_test"]
        else:
            src, tests = "d_fire_calval", ["d_fire_test", "d_fire_test_dedup"]
        assert f"{run}__to__{tests[0]}" in OPJ["cells"] and OPJ["cells"][f"{run}__to__{tests[0]}"]["source_calval"] == src, run
        cls = gt(src)[2] & model_classes(run, [src, tests[0]])
        s_max = img_max(run, src, cls)
        vals = {}
        for t in tests:
            if not (DETS / f"{run}__{t}.bbox.json").exists():
                continue
            vals[t] = float(stats.wasserstein_distance(s_max, img_max(run, t, cls)))
        null_split[run] = {"source_calval": src, "test": tests[0], "classes": sorted(cls), "W1_imgmax": vals[tests[0]]}
        for t, v in vals.items():
            if t != tests[0]:
                null_split_secondary[f"{run}__{src}__vs__{t}"] = v
    # --- in-domain null (ii): camera-disjoint folds on pooled pyro_sdis calval + caltest -------------------------
    folds = CAMJ["folds"]
    ids_cal, gt_cal, cls_p = gt("pyro_sdis_calval")
    ids_ct, gt_ct, _ = gt("pyro_sdis_caltest")
    cam_cal = {im["id"]: META[im["file_name"]]["camera"] for im in json.loads((GT / "pyro_sdis_calval.json").read_text(encoding="utf-8"))["images"]}
    cam_ct = {im["id"]: META[im["file_name"]]["camera"] for im in json.loads((GT / "pyro_sdis_caltest.json").read_text(encoding="utf-8"))["images"]}
    assert len(ids_cal) + len(ids_ct) == CAMJ["n_images"]
    null_cam = {}
    for run in [r for r in runs if "pyrosdis" in r]:
        cls = cls_p & model_classes(run, ["pyro_sdis_calval", "pyro_sdis_caltest"])
        m_cal, m_ct = img_max(run, "pyro_sdis_calval", cls), img_max(run, "pyro_sdis_caltest", cls)
        pooled = np.concatenate([m_cal, m_ct])
        cams = np.array([cam_cal[i] for i in ids_cal] + [cam_ct[i] for i in ids_ct])
        per_fold = []
        for f, fd in enumerate(folds):
            held = np.isin(cams, fd["cameras"])
            assert int(held.sum()) == fd["n_images"], (run, f, int(held.sum()), fd["n_images"])
            per_fold.append({"fold": f, "n_calibration_images": int((~held).sum()), "n_heldout_images": int(held.sum()),
                             "W1_imgmax": float(stats.wasserstein_distance(pooled[~held], pooled[held]))})
        null_cam[run] = {"classes": sorted(cls), "folds": per_fold, "max_over_folds": max(x["W1_imgmax"] for x in per_fold)}
    null_vals = [v["W1_imgmax"] for v in null_split.values()] + [x["W1_imgmax"] for v in null_cam.values() for x in v["folds"]]
    null_arr = np.array(null_vals)
    cutoff = float(null_arr.max())
    arg = max(list(null_split.items()), key=lambda kv: kv[1]["W1_imgmax"])
    argc = max(((r, x) for r, v in null_cam.items() for x in v["folds"]), key=lambda kv: kv[1]["W1_imgmax"])
    null = {"n_values": int(len(null_arr)), "n_split_values": len(null_split), "n_camera_fold_values": sum(len(v["folds"]) for v in null_cam.values()),
            "max": cutoff, "p95": float(np.percentile(null_arr, 95)), "median": float(np.median(null_arr)), "mean": float(null_arr.mean()), "min": float(null_arr.min()),
            "max_split": float(max(v["W1_imgmax"] for v in null_split.values())), "argmax_split": arg[0],
            "max_camera_fold": argc[1]["W1_imgmax"], "argmax_camera_fold": {"run": argc[0], "fold": argc[1]["fold"]},
            "secondary_split_values_not_in_null": null_split_secondary, "max_including_secondary": float(max([cutoff] + list(null_split_secondary.values()))),
            "split_values": null_split, "camera_fold_values": null_cam}
    # --- apply the cut-off to the 44 cross cells ------------------------------------------------------------------
    names = list(cross)
    w = np.array([cross[c]["W1_imgmax"] for c in names]); wpool = np.array([cross[c]["W1_pooled"] for c in names])
    drift = np.array([cross[c]["abs_drift_t_src_minus_t_oracle"] for c in names]); reg = np.array([cross[c]["regret_t_src"] for c in names])
    regm = np.array([cross[c]["regret_M1c"] for c in names])
    above = w > cutoff
    for c, a in zip(names, above):
        cross[c]["above_cutoff"] = bool(a)

    def side(mask):
        if mask.sum() == 0:
            return {"n": 0}
        return {"n": int(mask.sum()), "cells": [c for c, m in zip(names, mask) if m], "regret_t_src_mean": float(reg[mask].mean()), "regret_t_src_median": float(np.median(reg[mask])),
                "regret_t_src_max": float(reg[mask].max()), "abs_drift_mean": float(drift[mask].mean()), "abs_drift_median": float(np.median(drift[mask])), "abs_drift_max": float(drift[mask].max()),
                "regret_M1c_mean": float(regm[mask].mean()), "W1_imgmax_mean": float(w[mask].mean()), "W1_imgmax_min": float(w[mask].min()), "W1_imgmax_max": float(w[mask].max()),
                f"n_drift_gt_{DRIFT_CUT}": int((drift[mask] > DRIFT_CUT).sum()), f"n_regret_gt_{REGRET_CUT}": int((reg[mask] > REGRET_CUT).sum())}
    rule = {"cutoff_W1_imgmax": cutoff, "definition": "max in-domain W1 (24 own-split values + camera-disjoint fold values); no cross-domain cell used",
            "n_cross_cells": len(names), "n_above": int(above.sum()), "n_below_or_equal": int((~above).sum()), "above": side(above), "below": side(~above),
            "by_target_above": {t: int(sum(1 for c in names if cross[c]["target_dataset"] == t and cross[c]["above_cutoff"])) for t in sorted({cross[c]["target_dataset"] for c in names})},
            "by_target_n": {t: int(sum(1 for c in names if cross[c]["target_dataset"] == t)) for t in sorted({cross[c]["target_dataset"] for c in names})}}
    rule["mean_difference_above_minus_below"] = {"regret_t_src": (rule["above"].get("regret_t_src_mean", np.nan) - rule["below"].get("regret_t_src_mean", np.nan)) if above.sum() and (~above).sum() else None,
                                                 "abs_drift": (rule["above"].get("abs_drift_mean", np.nan) - rule["below"].get("abs_drift_mean", np.nan)) if above.sum() and (~above).sum() else None}
    y_drift, y_reg = drift > DRIFT_CUT, reg > REGRET_CUT
    rocs = {"W1_imgmax": {f"abs_drift_gt_{DRIFT_CUT}": {"auc": auc(w, y_drift), "n_pos": int(y_drift.sum()), "n_neg": int((~y_drift).sum())},
                          f"regret_t_src_gt_{REGRET_CUT}": {"auc": auc(w, y_reg), "n_pos": int(y_reg.sum()), "n_neg": int((~y_reg).sum())}},
            "W1_pooled": {f"abs_drift_gt_{DRIFT_CUT}": {"auc": auc(wpool, y_drift)}, f"regret_t_src_gt_{REGRET_CUT}": {"auc": auc(wpool, y_reg)}},
            "spearman": {"W1_imgmax_vs_abs_drift": [float(v) for v in stats.spearmanr(w, drift)], "W1_imgmax_vs_regret_t_src": [float(v) for v in stats.spearmanr(w, reg)]}}
    # confusion of the null cut-off against both targets
    for lab, y in ((f"abs_drift_gt_{DRIFT_CUT}", y_drift), (f"regret_t_src_gt_{REGRET_CUT}", y_reg)):
        tp, fp, fn, tn = int((above & y).sum()), int((above & ~y).sum()), int((~above & y).sum()), int((~above & ~y).sum())
        rule[f"confusion_{lab}"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
                                    "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)), "precision": tp / max(tp + fp, 1) if tp + fp else None}
    # --- leave-one-target-out for a FITTED cut-off ---------------------------------------------------------------
    tgt = np.array([cross[c]["target_dataset"] for c in names])
    loto = {}
    for lab, y in ((f"abs_drift_gt_{DRIFT_CUT}", y_drift), (f"regret_t_src_gt_{REGRET_CUT}", y_reg)):
        pred = np.zeros(len(names), dtype=bool); per_t = {}
        for t in sorted(set(tgt)):
            tr, te = tgt != t, tgt == t
            cands = np.unique(w[tr])
            best_c, best_ba = None, -1.0
            for cnd in cands:                                  # cut-off = value c; flag W1 > c (ties broken toward the smaller c)
                p = w[tr] > cnd
                tp, fn, tn, fp = (p & y[tr]).sum(), (~p & y[tr]).sum(), (~p & ~y[tr]).sum(), (p & ~y[tr]).sum()
                ba = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
                if ba > best_ba + 1e-12:
                    best_ba, best_c = ba, float(cnd)
            pred[te] = w[te] > best_c
            per_t[t] = {"fitted_cutoff": best_c, "train_balanced_accuracy": float(best_ba), "n_test": int(te.sum()), "n_test_pos": int(y[te].sum()),
                        "test_accuracy": float(np.mean(pred[te] == y[te])), "test_n_correct": int((pred[te] == y[te]).sum())}
        tp, fp, fn, tn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum()), int((~pred & ~y).sum())
        loto[lab] = {"per_heldout_target": per_t, "pooled": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / len(names),
                                                            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
                                                            "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))}}
    return {"definition": "W1 = scipy.stats.wasserstein_distance between per-image maximum scores (op.match per_img max over the evaluated classes); cross cells: source calval vs target calval (unlabelled)",
            "pilot_reproduction_max_abs_diff_W1_imgmax": max_w_diff, "cross_cells": cross, "in_domain_null": null, "null_cutoff_rule": rule,
            "roc": rocs, "leave_one_target_out_fitted_cutoff": loto, "seconds": round(time.time() - t0, 1)}


# =============================================================================== PART C
def part_c():
    out = {"source": "label_free_threshold_refine.json summary.groups.all (+ cells) and label_free_threshold_pilot.json summary.groups.all (+ cells)", "rules": {}}
    for src_name, J in (("refine", REFINE), ("pilot", PILOT)):
        G = J["summary"]["groups"]["all"]
        for m, v in G.items():
            if src_name == "pilot" and m in REFINE["summary"]["groups"]["all"]:
                continue                                       # M0 / D25 / M1c / ORC are taken from the refine artefact
            f1 = lambda c, mm: (J["cells"][c]["methods"][mm]["f1"] if src_name == "refine" else J["cells"][c]["methods"][mm]["box"]["f1"])
            beats = worse = ties = 0; tie_cells = []
            for c in J["cells"]:
                d = f1(c, m) - f1(c, "M0_t_src")
                if d > 1e-12: beats += 1
                elif d < -1e-12: worse += 1
                else: ties += 1; tie_cells.append(c)
            assert (beats, worse) == (v["n_beats_t_src"], v["n_worse_than_t_src"]), (m, beats, worse, v)
            assert beats + worse + ties == v["n_cells"]
            out["rules"][m] = {"artefact": src_name, "n_cells": v["n_cells"], "beats_t_src": v["n_beats_t_src"], "worse_than_t_src": v["n_worse_than_t_src"],
                               "ties_or_undefined": v["n_cells"] - v["n_beats_t_src"] - v["n_worse_than_t_src"], "tie_cells": tie_cells,
                               "regret_mean": v["regret_mean"], "regret_median": v["regret_median"],
                               "keys": f"{src_name}.summary.groups.all.{m}.{{n_cells,n_beats_t_src,n_worse_than_t_src,regret_mean}}"}
    return out


# =============================================================================== main
def main():
    t_start = time.time()
    A, Bp, C = part_a(), part_b(), part_c()
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "protocol": __doc__, "iou": op.IOU, "part_A_clean_split": A, "part_B_shift_rule": Bp, "part_C_table4_tallies": C,
           "seconds_total": round(time.time() - t_start, 1)}
    (HERE / "label_free_clean_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    f3 = lambda x: "" if x is None else f"{x:.3f}"
    cis = lambda c: "" if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]"
    L = ["# Label-free threshold transfer — clean-split check, shift-rule, Table-4 tallies (machine-written by label_free_clean_check.py)", "",
         f"Created {out['created']}. Keys quoted as part_A_clean_split / part_B_shift_rule / part_C_table4_tallies of label_free_clean_check.json.", "",
         "## Part A — kept-rate rule with the de-duplicated D-Fire calibration split as pool + oracle set, clean smoke-only test", "",
         f"Mapping check: {A['mapping_check']}. Official-split reproduction of t_M1c / t_ORC vs the refine artefact: {A['reproduction_official_split']['n_cells']} cells, "
         f"max |dt| M1c = {A['reproduction_official_split']['max_abs_diff_M1c']:.2e}, ORC = {A['reproduction_official_split']['max_abs_diff_ORC']:.2e}, max |dF1| = {A['reproduction_official_split']['max_abs_diff_f1']:.2e} (passed); "
         f"cells without a refine counterpart (official numbers computed here only): {A['reproduction_official_split']['cells_without_refine_counterpart']}. "
         f"Not evaluable: {A['not_evaluable'] or 'none'}. Bootstrap: B = {B}, seed {SEED}, clean-test images.", "",
         "| cell | t_src | t_M1c clean / official | t_ORC clean / official | F1 t_src clean [CI] / official | F1 M1c clean [CI] / official | F1 ORC clean [CI] / official | regret M1c clean [CI] / official | gap closed clean [CI] / official | F1 official t_M1c on clean test |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c, e in A["cells"].items():
        m, o = e["methods"], e["official_split"]
        gc = m["M1c_kept_rate"]["gap_closed"]
        L.append(f"| {c} | {f3(m['M0_t_src']['threshold'])} | {f3(m['M1c_kept_rate']['threshold'])} / {f3(o['t_M1c'])} | {f3(m['ORC_target_calval']['threshold'])} / {f3(o['t_ORC'])} | "
                 f"{f3(m['M0_t_src']['f1'])} {cis(m['M0_t_src']['f1_ci95'])} / {f3(o['f1_t_src'])} | {f3(m['M1c_kept_rate']['f1'])} {cis(m['M1c_kept_rate']['f1_ci95'])} / {f3(o['f1_M1c'])} | "
                 f"{f3(m['ORC_target_calval']['f1'])} {cis(m['ORC_target_calval']['f1_ci95'])} / {f3(o['f1_ORC'])} | "
                 f"{f3(m['M1c_kept_rate']['regret_vs_oracle'])} {cis(m['M1c_kept_rate']['regret_vs_oracle_ci95'])} / {f3(o['regret_M1c'])} | "
                 f"{f3(gc['point'])} {cis(gc['ci95'])}{' (unstable)' if gc['unstable'] else ''} / {f3(o['gap_closed_M1c'])} | {f3(m['M1c_official_threshold_on_clean_test']['f1'])} |")
    L += ["", "| group | n | F1 t_src clean / off | F1 M1c clean / off | F1 ORC clean / off | regret t_src clean / off | regret M1c clean [CI img] [CI cell] / off | gap closed M1c clean / off | M1c beats/worse t_src (clean) | regret D25 clean / off |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g, s in A["groups"].items():
        o = s["official_split_means"]
        L.append(f"| {g} | {s['n_cells']} | {f3(s['M0_t_src']['f1_mean'])} / {f3(o['f1_t_src_mean'])} | {f3(s['M1c_kept_rate']['f1_mean'])} / {f3(o['f1_M1c_mean'])} | "
                 f"{f3(s['ORC_target_calval']['f1_mean'])} / {f3(o['f1_ORC_mean'])} | {f3(s['M0_t_src']['regret_mean'])} / {f3(o['regret_t_src_mean'])} | "
                 f"{f3(s['M1c_kept_rate']['regret_mean'])} {cis(s['M1c_kept_rate']['regret_mean_ci95_image_bootstrap'])} {cis(s['M1c_kept_rate']['regret_mean_ci95_cell_bootstrap'])} / {f3(o['regret_M1c_mean'])} | "
                 f"{f3(s['M1c_kept_rate']['gap_closed_mean'])} / {f3(o['gap_closed_M1c_mean'])} | {s['M1c_kept_rate']['n_beats_t_src']}/{s['M1c_kept_rate']['n_worse_than_t_src']} | "
                 f"{f3(s['D25_default']['regret_mean'])} / {f3(o['regret_D25_mean'])} |")
    n_, r_, ro = Bp["in_domain_null"], Bp["null_cutoff_rule"], Bp["roc"]
    L += ["", "## Part B — shift diagnostic W1 (per-image max) and an in-domain null cut-off", "",
          f"Pilot W1 reproduced (max |dW1| = {Bp['pilot_reproduction_max_abs_diff_W1_imgmax']:.2e}). In-domain null: n = {n_['n_values']} ({n_['n_split_values']} own-split + {n_['n_camera_fold_values']} camera-fold values); "
          f"max = {n_['max']:.4f} (split max {n_['max_split']:.4f} at {n_['argmax_split']}; camera-fold max {n_['max_camera_fold']:.4f} at {n_['argmax_camera_fold']}), p95 = {n_['p95']:.4f}, median = {n_['median']:.4f}; "
          f"max including the secondary dedup-test values = {n_['max_including_secondary']:.4f}.", "",
          f"Cut-off = {r_['cutoff_W1_imgmax']:.4f}: {r_['n_above']} of {r_['n_cross_cells']} cross cells above, {r_['n_below_or_equal']} below (by target above/n: "
          + ", ".join(f"{t} {r_['by_target_above'][t]}/{r_['by_target_n'][t]}" for t in r_['by_target_n']) + ").", "",
          "| side | n | regret t_src mean | median | max | abs drift mean | median | max | regret M1c mean | W1 min-max | n drift>0.1 | n regret>0.02 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sd in ("above", "below"):
        s = r_[sd]
        if s["n"]:
            L.append(f"| {sd} | {s['n']} | {f3(s['regret_t_src_mean'])} | {f3(s['regret_t_src_median'])} | {f3(s['regret_t_src_max'])} | {f3(s['abs_drift_mean'])} | {f3(s['abs_drift_median'])} | {f3(s['abs_drift_max'])} | "
                     f"{f3(s['regret_M1c_mean'])} | {s['W1_imgmax_min']:.3f}-{s['W1_imgmax_max']:.3f} | {s[f'n_drift_gt_{DRIFT_CUT}']} | {s[f'n_regret_gt_{REGRET_CUT}']} |")
        else:
            L.append(f"| {sd} | 0 | | | | | | | | | | |")
    L += ["", f"ROC AUC of W1_imgmax: |drift| > {DRIFT_CUT}: {f3(ro['W1_imgmax'][f'abs_drift_gt_{DRIFT_CUT}']['auc'])} (pos {ro['W1_imgmax'][f'abs_drift_gt_{DRIFT_CUT}']['n_pos']}/neg {ro['W1_imgmax'][f'abs_drift_gt_{DRIFT_CUT}']['n_neg']}); "
          f"regret(t_src) > {REGRET_CUT}: {f3(ro['W1_imgmax'][f'regret_t_src_gt_{REGRET_CUT}']['auc'])} (pos {ro['W1_imgmax'][f'regret_t_src_gt_{REGRET_CUT}']['n_pos']}/neg {ro['W1_imgmax'][f'regret_t_src_gt_{REGRET_CUT}']['n_neg']}). "
          f"W1_pooled: {f3(ro['W1_pooled'][f'abs_drift_gt_{DRIFT_CUT}']['auc'])} / {f3(ro['W1_pooled'][f'regret_t_src_gt_{REGRET_CUT}']['auc'])}. "
          f"Spearman W1_imgmax vs |drift| rho = {ro['spearman']['W1_imgmax_vs_abs_drift'][0]:.3f}, vs regret rho = {ro['spearman']['W1_imgmax_vs_regret_t_src'][0]:.3f}.", ""]
    for lab in (f"abs_drift_gt_{DRIFT_CUT}", f"regret_t_src_gt_{REGRET_CUT}"):
        cf, lo = r_[f"confusion_{lab}"], Bp["leave_one_target_out_fitted_cutoff"][lab]["pooled"]
        L.append(f"- null cut-off vs {lab}: tp {cf['tp']} fp {cf['fp']} fn {cf['fn']} tn {cf['tn']}, sens {cf['sensitivity']:.3f}, spec {cf['specificity']:.3f}, bal-acc {cf['balanced_accuracy']:.3f}; "
                 f"leave-one-target-out fitted cut-off (balanced accuracy): pooled acc {lo['accuracy']:.3f}, sens {lo['sensitivity']:.3f}, spec {lo['specificity']:.3f} "
                 f"(fitted cut-offs: " + ", ".join(f"{t} {v['fitted_cutoff']:.3f}" for t, v in Bp['leave_one_target_out_fitted_cutoff'][lab]['per_heldout_target'].items()) + ")")
    L += ["", "| cell | target | W1 imgmax | above | abs drift | regret t_src | regret M1c |", "|---|---|---|---|---|---|---|"]
    for c, e in sorted(Bp["cross_cells"].items(), key=lambda kv: -kv[1]["W1_imgmax"]):
        L.append(f"| {c} | {e['target_dataset']} | {e['W1_imgmax']:.4f} | {'yes' if e['above_cutoff'] else 'no'} | {f3(e['abs_drift_t_src_minus_t_oracle'])} | {f3(e['regret_t_src'])} | {f3(e['regret_M1c'])} |")
    L += ["", "| in-domain null (own split) | source calval | test | W1 imgmax |", "|---|---|---|---|"]
    for r, v in n_["split_values"].items():
        L.append(f"| {r} | {v['source_calval']} | {v['test']} | {v['W1_imgmax']:.4f} |")
    L += ["", "| in-domain null (camera folds) | fold 0 | fold 1 | fold 2 | fold 3 | max |", "|---|---|---|---|---|---|"]
    for r, v in n_["camera_fold_values"].items():
        L.append(f"| {r} | " + " | ".join(f"{x['W1_imgmax']:.4f}" for x in v["folds"]) + f" | {v['max_over_folds']:.4f} |")
    L += ["", "## Part C — Table-4 tallies (n = beats + worse + ties/undefined)", "", "| rule | artefact | n | beats t_src | worse | ties/undefined | regret mean | regret median |", "|---|---|---|---|---|---|---|---|"]
    for m, v in C["rules"].items():
        L.append(f"| {m} | {v['artefact']} | {v['n_cells']} | {v['beats_t_src']} | {v['worse_than_t_src']} | {v['ties_or_undefined']} | {f3(v['regret_mean'])} | {f3(v['regret_median'])} |")
    (HERE / "label_free_clean_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("artefact ->", HERE / "label_free_clean_check.json", "and .md", f"({out['seconds_total']} s)")


if __name__ == "__main__":
    main()
