"""Decompose "miscalibration under shift" into threshold choice vs confidence map.

The toolbox evaluates calibration on detections that survive an LRP-optimal
threshold fitted on the calibration split; under dataset shift the SOURCE-
fitted threshold is itself mis-set for the target. For every cross-dataset
cell this evaluates the grid

    threshold origin : raw-score LRP-optimal threshold fitted on SOURCE calval
                       | on TARGET calval   (per class, keyed by category id)
    confidence map   : none | Platt / isotonic fitted on SOURCE | on TARGET

with NO second (post-calibration) threshold, so the threshold axis is purely
the raw-score cut and the map axis purely the confidence transform. Reports
LaECE_0, D-ECE, LRP (+FN/FP) and detections kept for each condition.
Check: the (thr=source, map=none) condition must reproduce the forward cell's
identity LaECE_0 (to 1e-6): the identity map's effective cut is
max(pre-threshold, operating threshold), both fitted on the same calval. The toolbox's coupled conventions remain in
calibration/<cell>.json and <cell>__repair.json.
Writes analysis/threshold_decomposition.json.
"""
import contextlib
import copy
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
GT, DETS, CAL = HERE / "coco_gt", HERE / "detections", HERE / "calibration"
TARGET_CALVAL = {"d_fire": "d_fire_calval", "pyro_sdis": "pyro_sdis_calval", "thesis": "thesis_calval"}
MAPS = ["platt_scaling", "isotonic_regression"]


def _q(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def cats_of(gt_path):
    return [c["id"] for c in json.loads(Path(gt_path).read_text(encoding="utf-8"))["categories"]]


def load_dets(path, cats):
    return [d for d in json.loads(Path(path).read_text(encoding="utf-8")) if d["category_id"] in cats]


def fit(gt_path, dets, m, test_gt):
    """Return (cm, model, {category_id: raw threshold}) for a calibrator fitted on gt_path."""
    from detection_calibration.DetectionCalibration import DetectionCalibration
    tmp = HERE / "tmp_calib_inputs"; tmp.mkdir(exist_ok=True)
    fp = tmp / f"decomp_fit_{Path(gt_path).stem}_{m}.json"
    fp.write_text(json.dumps(dets), encoding="utf-8")
    cm = DetectionCalibration(gt_path, test_gt)
    model, thr = _q(cm.fit, str(fp), calibrator_type=m)
    classes = cm.calibration_scheme.dataset_classes
    # toolbox applies pre-threshold then operating threshold sequentially; for
    # the identity map both act on raw scores, so the effective cut is their max
    thr_raw = {int(c): float(max(a, b)) for c, a, b in zip(classes, np.atleast_1d(thr[0]), np.atleast_1d(thr[1]))}
    return cm, model, thr_raw


def apply(dets, thr_raw, cm=None, model=None, m=None):
    kept = [copy.deepcopy(d) for d in dets if d["score"] >= thr_raw.get(d["category_id"], np.inf)]
    if model is not None:
        kept = cm.calibration_scheme.calibrate(kept, m, model, cm.calibration_scheme.dataset_classes)
        for d in kept:
            d["score"] = float(np.clip(d["score"], 0, 1))
    return kept


def main():
    import calib_metrics
    out = {"grid": "threshold in {source,target} x map in {none, platt/iso fitted on source, platt/iso fitted on target}",
           "cells": {}}
    for art in sorted(CAL.glob("*.json")):
        d = json.loads(art.read_text(encoding="utf-8"))
        cell = d["cell"]
        if cell.endswith("__repair"):
            continue
        inp = d["inputs"]
        run = cell.split("__to__")[0]
        src_tag, test_tag = Path(inp["calval-gt"]).stem, Path(inp["test-gt"]).stem
        tgt_ds = next((k for k in TARGET_CALVAL if test_tag.startswith(k)), None)
        if tgt_ds is None or src_tag == TARGET_CALVAL[tgt_ds]:
            continue
        tgt_calval = TARGET_CALVAL[tgt_ds]
        tgt_dets_p = DETS / f"{run}__{tgt_calval}.bbox.json"
        if not tgt_dets_p.exists():
            print(f"[skip] {cell}: no {tgt_dets_p.name}"); continue
        test_gt = inp["test-gt"]; tgt_gt = str(GT / f"{tgt_calval}.json")
        src_cats, tgt_cats, test_cats = cats_of(inp["calval-gt"]), cats_of(tgt_gt), cats_of(test_gt)
        s_dets = load_dets(inp["calval-dets"], src_cats)
        t_dets = load_dets(str(tgt_dets_p), tgt_cats)
        x_dets = load_dets(inp["test-dets"], test_cats)
        fits = {("source", "identity"): fit(inp["calval-gt"], s_dets, "identity", test_gt),
                ("target", "identity"): fit(tgt_gt, t_dets, "identity", test_gt)}
        for m in MAPS:
            fits[("source", m)] = fit(inp["calval-gt"], s_dets, m, test_gt)
            fits[("target", m)] = fit(tgt_gt, t_dets, m, test_gt)
        res = {}
        for thr_origin in ("source", "target"):
            thr_raw = fits[(thr_origin, "identity")][2]
            conds = [("none", None)] + [(o, m) for o in ("source", "target") for m in MAPS]
            for map_origin, m in conds:
                if m is None:
                    dets = apply(x_dets, thr_raw)
                else:
                    cm, model, _ = fits[(map_origin, m)]
                    dets = apply(x_dets, thr_raw, cm, model, m)
                key = f"thr={thr_origin}|map={map_origin}" + ("" if m is None else f":{m.split('_')[0]}")
                if not dets:   # every detection cut by this threshold: nothing to calibrate
                    res[key] = {"LaECE_0": None, "D_ECE": None, "lrp": 1.0, "lrp_fn": 1.0, "lrp_fp": None,
                                "n_dets_kept": 0, "thr_raw": thr_raw, "note": "no detections survive the threshold"}
                    continue
                e = calib_metrics.evaluate(test_gt, dets)
                res[key] = {"LaECE_0": e["LaECE_0"], "D_ECE": e["D_ECE"], "lrp": e["lrp"], "lrp_fn": e["lrp_fn"],
                            "lrp_fp": e["lrp_fp"], "n_dets_kept": len(dets), "thr_raw": thr_raw}
        ref = d["results"]["identity"]["LaECE_0"]
        got = res["thr=source|map=none"]["LaECE_0"]
        res["check_reproduces_forward_identity"] = {"forward_cell": ref, "here": got, "ok": abs(ref - got) < 1e-6}
        out["cells"][cell] = {"run": run, "source_calval": src_tag, "target_calval": tgt_calval, "test": test_tag, "results": res}
        print(cell, "check", res["check_reproduces_forward_identity"]["ok"],
              {k: (round(v["LaECE_0"] * 100, 1) if v.get("LaECE_0") is not None else None) for k, v in res.items() if "LaECE_0" in v},
              "FN:", {k: round(v["lrp_fn"] * 100, 1) for k, v in res.items() if "map=none" in k}, flush=True)
    (HERE / "threshold_decomposition.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("artefact ->", HERE / "threshold_decomposition.json")


if __name__ == "__main__":
    main()
