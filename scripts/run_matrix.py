"""Drive the transfer-matrix cells: export predictions + calibration battery.

For each trained run and each declared target, this: (1) exports predictions
on the target's calval+test GT (skipping exports that already exist), then
(2) runs the calibration battery (skipping cells whose artefact exists).
Idempotent — safe to rerun after each training run lands.

Matrix cells are declared in CELLS below and grow as runs/datasets land.
GPU note: exports run inference — only invoke while the GPU is free.

Usage: .venv/Scripts/python analysis/run_matrix.py [--dry-run]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
GT = HERE / "coco_gt"
DETS = HERE / "detections"

# (run_name, source_calval_tag, [target tags for eval])
# each target tag T needs GT/<T>.json; the source calval is where calibrators fit.
CELLS = [
    ("v8s_pyrosdis_s3407", "pyro_sdis_calval",
     ["pyro_sdis_caltest", "d_fire_test", "d_fire_test_smokeonly"]),
    ("v8s_dfire_s3407", "d_fire_calval", ["d_fire_test", "pyro_sdis_caltest"]),
    ("y11s_dfire_s3407", "d_fire_calval", ["d_fire_test", "pyro_sdis_caltest"]),
    ("y11s_pyrosdis_s3407", "pyro_sdis_calval",
     ["pyro_sdis_caltest", "d_fire_test", "d_fire_test_smokeonly"]),
    ("rtdetrl_dfire_s3407", "d_fire_calval", ["d_fire_test", "pyro_sdis_caltest"]),
    # THIRD TARGET DOMAIN (2026-09-03): author's thesis set (Public Domain,
    # ground/UAV close-range mixed; never trained on). fire+smoke models ->
    # thesis_test; smoke-only models -> thesis_test_smokeonly (+ full for record).
    ("v8s_dfire_s3407", "d_fire_calval", ["thesis_test"]),
    ("y11s_dfire_s3407", "d_fire_calval", ["thesis_test"]),
    ("rtdetrl_dfire_s3407", "d_fire_calval", ["thesis_test"]),
    ("v8s_pyrosdis_s3407", "pyro_sdis_calval", ["thesis_test", "thesis_test_smokeonly"]),
    ("y11s_pyrosdis_s3407", "pyro_sdis_calval", ["thesis_test", "thesis_test_smokeonly"]),
    # FOURTH DOMAIN (2026-09-03): FIgLib box-labelled subset (9 sequences, 609
    # frames, smoke boxes from HPWREN-BB CSVs) as a detection-level target.
    ("v8s_pyrosdis_s3407", "pyro_sdis_calval", ["figlib_bb"]),
    ("y11s_pyrosdis_s3407", "pyro_sdis_calval", ["figlib_bb"]),
    ("v8s_dfire_s3407", "d_fire_calval", ["figlib_bb"]),
    ("y11s_dfire_s3407", "d_fire_calval", ["figlib_bb"]),
    ("rtdetrl_dfire_s3407", "d_fire_calval", ["figlib_bb"]),
    # REPAIR experiment: same runs, calibrators fitted on the TARGET domain's
    # calval (small in-target sample) instead of the source — does target-side
    # post-hoc repair fix what source-fitted repair cannot?
    ("v8s_pyrosdis_s3407", "d_fire_calval", ["d_fire_test_smokeonly"], "repair"),
    ("y11s_pyrosdis_s3407", "d_fire_calval", ["d_fire_test_smokeonly"], "repair"),
    ("v8s_dfire_s3407", "pyro_sdis_calval", ["pyro_sdis_caltest"], "repair"),
    ("y11s_dfire_s3407", "pyro_sdis_calval", ["pyro_sdis_caltest"], "repair"),
    ("rtdetrl_dfire_s3407", "pyro_sdis_calval", ["pyro_sdis_caltest"], "repair"),
    ("v8s_dfire_s3407", "thesis_calval", ["thesis_test"], "repair"),
    ("y11s_dfire_s3407", "thesis_calval", ["thesis_test"], "repair"),
    ("rtdetrl_dfire_s3407", "thesis_calval", ["thesis_test"], "repair"),
    ("v8s_pyrosdis_s3407", "thesis_calval", ["thesis_test_smokeonly"], "repair"),
    ("y11s_pyrosdis_s3407", "thesis_calval", ["thesis_test_smokeonly"], "repair"),
]

# AUTO cells (2026-09-03): any finished run not named in CELLS gets the
# standard target set for its source dataset — covers rtdetrl_pyrosdis, the
# seed-1337 replicates and the n/m scale runs without editing this file.
AUTO = {
    "dfire": {"calval": "d_fire_calval",
              "targets": ["d_fire_test", "pyro_sdis_caltest", "thesis_test", "figlib_bb"],
              "repair": [("pyro_sdis_calval", "pyro_sdis_caltest"), ("thesis_calval", "thesis_test")]},
    "dfiredd": {"calval": "d_fire_calval_dedup",
                "targets": ["d_fire_test_dedup", "d_fire_test", "pyro_sdis_caltest", "thesis_test", "figlib_bb"],
                "repair": [("pyro_sdis_calval", "pyro_sdis_caltest"), ("thesis_calval", "thesis_test")]},
    "dfiresub": {"calval": "d_fire_calval",
                 "targets": ["d_fire_test_dedup", "d_fire_test", "pyro_sdis_caltest", "thesis_test", "figlib_bb"],
                 "repair": [("pyro_sdis_calval", "pyro_sdis_caltest"), ("thesis_calval", "thesis_test")]},
    "pyrosdis": {"calval": "pyro_sdis_calval",
                 "targets": ["pyro_sdis_caltest", "d_fire_test", "d_fire_test_smokeonly", "thesis_test", "thesis_test_smokeonly", "figlib_bb"],
                 "repair": [("d_fire_calval", "d_fire_test_smokeonly"), ("thesis_calval", "thesis_test_smokeonly")]},
}

def all_cells():
    named = {c[0] for c in CELLS}
    cells = list(CELLS)
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name in named or not (d / "weights" / "best.pt").exists():
            continue
        src = "dfiresub" if "_dfiresub_" in d.name else "dfiredd" if "_dfiredd_" in d.name else "dfire" if "_dfire_" in d.name else "pyrosdis" if "_pyrosdis_" in d.name else None
        if src is None or not (d / "results.png").exists():   # results.png = training finished
            continue
        a = AUTO[src]
        cells.append((d.name, a["calval"], a["targets"]))
        for cv, tg in a["repair"]:
            cells.append((d.name, cv, [tg], "repair"))
    return cells


def sh(cmd, dry):
    print("+", " ".join(map(str, cmd)))
    if not dry:
        r = subprocess.run(list(map(str, cmd)))
        if r.returncode != 0:
            sys.exit(f"FAILED: {cmd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="0", help="export GPU; '1' = T600 while device 0 trains")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--only", default=None, help="substring filter on run name")
    args = ap.parse_args()
    py = sys.executable

    for entry in all_cells():
        run, calval, targets = entry[0], entry[1], entry[2]
        suffix = f"__{entry[3]}" if len(entry) > 3 else ""
        if args.only and args.only not in run:
            continue
        weights = RUNS / run / "weights" / "best.pt"
        if not weights.exists():
            print(f"[skip] {run}: no best.pt yet")
            continue
        for tag in [calval] + targets:
            gt = GT / f"{tag}.json"
            if not gt.exists():
                print(f"[skip] {run}->{tag}: GT missing")
                continue
            det = DETS / f"{run}__{tag}.bbox.json"
            if det.exists():
                print(f"[have] {det.name}")
            else:
                sh([py, HERE / "export_predictions.py", "--weights", weights,
                    "--gt", gt, "--out", det, "--device", args.device,
                    "--batch", args.batch], args.dry_run)
        for tag in targets:
            det_val = DETS / f"{run}__{calval}.bbox.json"
            det_tst = DETS / f"{run}__{tag}.bbox.json"
            cell = f"{run}__to__{tag}{suffix}"
            art = HERE / "calibration" / f"{cell}.json"
            if art.exists():
                print(f"[have] {art.name}")
                continue
            if not (det_val.exists() or args.dry_run) or not (det_tst.exists() or args.dry_run):
                print(f"[skip] {cell}: detections missing")
                continue
            sh([py, HERE / "run_calibration.py", "--calval-gt", GT / f"{calval}.json",
                "--calval-dets", det_val, "--test-gt", GT / f"{tag}.json",
                "--test-dets", det_tst, "--cell", cell, "--plots"], args.dry_run)


if __name__ == "__main__":
    main()
