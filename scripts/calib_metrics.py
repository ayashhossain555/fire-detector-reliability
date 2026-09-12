"""Metrics wrapper around detection_calibration internals that RETURNS values.

The toolbox's evaluate_calibration() only prints; the hub's verification rule
needs machine-written numbers. This mirrors its computation (pinned commit
bf59c92) and returns a dict: LRP components, LaECE_0, LaACE_0, D-ECE, plus
the settings used. Reliability-diagram plotting is delegated to the toolbox
(saves into cwd) when plot_dir is given.
"""

import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np
from detection_calibration.coco_calibration import CalibrationCOCO
from pycocotools.coco import COCO


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def evaluate(test_annotations: str, detections, tau: float = 0.0,
             bin_count: int = 25, max_dets: int = 100, plot_dir: str | None = None) -> dict:
    """detections: list of COCO-format dicts (already calibrated or raw)."""
    # LaECE + LRP (mirrors coco_calibration.evaluate_calibration lines 597-646)
    ev = CalibrationCOCO(test_annotations, test_annotations, "bbox", 25, tau, False, False, max_dets)
    ev.cocoDt = _quiet(COCO(test_annotations).loadRes, detections)
    _quiet(ev.evaluate); _quiet(ev.prepare_input); _quiet(ev.compute_single_errors)
    laece = float(ev.accumulate_errors())
    if plot_dir:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        try:
            os.chdir(plot_dir)
            ev.plot_reliability_diagram(laece, cl=-1, fontsize=22)
        finally:
            os.chdir(cwd)

    ev2 = CalibrationCOCO(test_annotations, test_annotations, "bbox", bin_count, tau, False, True, max_dets)
    ev2.cocoDt = _quiet(COCO(test_annotations).loadRes, detections)
    _quiet(ev2.evaluate); _quiet(ev2.prepare_input); _quiet(ev2.compute_single_errors)
    _quiet(ev2.compute_LRP)
    laace = float(ev2.accumulate_errors())
    lrp = {k: float(np.nanmean(ev2.lrps[k])) for k in ("lrp", "lrp_loc", "lrp_fp", "lrp_fn")}

    # D-ECE (their D-ECE convention: bbox, 10 bins, tau 0.5)
    ev3 = CalibrationCOCO(test_annotations, test_annotations, "bbox", 10, 0.5, True, False, max_dets)
    ev3.cocoDt = _quiet(COCO(test_annotations).loadRes, detections)
    _quiet(ev3.evaluate); _quiet(ev3.prepare_input); _quiet(ev3.compute_single_errors)
    _quiet(ev3.compute_LRP)
    dece = float(ev3.accumulate_errors())

    return {"LaECE_0": laece, "LaACE_0": laace, "D_ECE": dece, **lrp,
            "settings": {"tau": tau, "bin_count": bin_count, "max_dets": max_dets,
                         "dece_convention": "bbox/10bins/tau0.5",
                         "toolbox_commit": "bf59c92"}}
