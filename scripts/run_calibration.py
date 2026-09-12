"""Run the post-hoc calibration battery for one (model, source, target) cell.

Per TRAINING-PROTOCOL.md: calibrators are fitted on the SOURCE calval
detections; evaluated on the TARGET test detections. `identity` = the
uncalibrated baseline. Writes one JSON artefact per cell into
analysis/calibration/ — every paper number traces here.

Usage:
  .venv/Scripts/python analysis/run_calibration.py \
      --calval-gt analysis/coco_gt/pyro_sdis_calval.json \
      --calval-dets analysis/detections/v8s_pyrosdis__pyro_sdis_calval.bbox.json \
      --test-gt analysis/coco_gt/pyro_sdis_caltest.json \
      --test-dets analysis/detections/v8s_pyrosdis__pyro_sdis_caltest.bbox.json \
      --cell v8s_pyrosdis__to__pyro_sdis
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
CALIBRATORS = ["identity", "temperature_scaling", "platt_scaling", "isotonic_regression"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calval-gt", required=True)
    ap.add_argument("--calval-dets", required=True)
    ap.add_argument("--test-gt", required=True)
    ap.add_argument("--test-dets", required=True)
    ap.add_argument("--cell", required=True, help="artefact name: <run>__to__<target>")
    ap.add_argument("--plots", action="store_true", help="save reliability diagrams")
    args = ap.parse_args()

    from detection_calibration.DetectionCalibration import DetectionCalibration
    import calib_metrics

    out = {"cell": args.cell,
           "inputs": {k: getattr(args, k.replace("-", "_")) for k in
                      ("calval-gt", "calval-dets", "test-gt", "test-dets")},
           "results": {}}
    # Class-space guard (2026-09-03): a fire+smoke model calibrated on a
    # smoke-only calval crashes the toolbox (category index lookup). Drop
    # detections whose category is absent from the calval GT — they would be
    # unevaluable there anyway — and record how many (TOOLBOX-PATCHES.md).
    calval_cats = {c["id"] for c in json.loads(Path(args.calval_gt).read_text(encoding="utf-8"))["categories"]}
    tmp = HERE / "tmp_calib_inputs"; tmp.mkdir(exist_ok=True)
    def filtered(path, tag):
        dets = json.loads(Path(path).read_text(encoding="utf-8"))
        keep = [d for d in dets if d["category_id"] in calval_cats]
        out["inputs"][f"{tag}_dets_dropped_offclass"] = len(dets) - len(keep)
        if len(keep) == len(dets):
            return path
        fp = tmp / f"{args.cell}__{tag}.json"; fp.write_text(json.dumps(keep), encoding="utf-8")
        return str(fp)
    calval_dets_path, test_dets_path = filtered(args.calval_dets, "calval"), filtered(args.test_dets, "test")
    for cal_type in CALIBRATORS:
        cm = DetectionCalibration(args.calval_gt, args.test_gt)
        calibrator, thresholds = cm.fit(calval_dets_path, calibrator_type=cal_type)
        cal_dets = cm.transform(test_dets_path, calibrator, thresholds)
        plot_dir = (HERE / "figures_calibration" / f"{args.cell}__{cal_type}") if args.plots else None
        m = calib_metrics.evaluate(args.test_gt, cal_dets,
                                   plot_dir=str(plot_dir) if plot_dir else None)
        import numpy as np
        m["thresholds"] = {"pre": [float(v) for v in np.atleast_1d(thresholds[0])],
                           "operating": [float(v) for v in np.atleast_1d(thresholds[1])]}
        out["results"][cal_type] = m
        print(f"[{args.cell}] {cal_type}: LaECE_0 {m['LaECE_0']*100:.1f} | "
              f"D-ECE {m['D_ECE']*100:.1f} | LRP {m['lrp']*100:.1f}")

    dest = HERE / "calibration" / f"{args.cell}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("artefact ->", dest)


if __name__ == "__main__":
    main()
