"""Rebuild dedup_training_comparison.json from saved artefacts.

Leakage-at-training comparison for D-Fire: FULL training set (15,499 images,
seeds 3407 and 1337) vs the size-matched random SUBSET (8,316, still leaky)
vs the DEDUP set (8,316, test-clean). The JSON was first assembled in-session
on 2026-09-03; this script is its saved, re-runnable writer so that every
value traces to an artefact:

  mAP50 (clean / official / leaky test images)  <- dfire_leakage.json cells.<run>__to__d_fire_test_dedup
  val mAP50, best epoch                          <- runs_meta/<run>.json
  LaECE / Platt LaECE / LRP-FN per target        <- matrix_summary.json cells.<run>__to__<target> metrics

Output: analysis/dedup_training_comparison.json (same schema as before, plus
per-run mAP50_leaky_images and a 'sources' block).
"""
import json
from pathlib import Path

A = Path(__file__).resolve().parent
leak = json.load(open(A / "dfire_leakage.json", encoding="utf-8"))["cells"]
matrix = json.load(open(A / "matrix_summary.json", encoding="utf-8"))["cells"]

RUNS = [
    "v8s_dfire_s3407", "v8s_dfire_s1337", "v8s_dfiresub_s3407", "v8s_dfiredd_s3407",
    "y11s_dfire_s3407", "y11s_dfire_s1337", "y11s_dfiredd_s3407",
]
TARGETS = ["d_fire_test_dedup", "d_fire_test", "pyro_sdis_caltest", "thesis_test"]

out = {
    "note": ("leakage-at-training comparison: full (15,499) vs size-matched random subset "
             "(8,316, leaky) vs de-duplicated (8,316, test-clean); mAP50 via pycocotools "
             "(dfire_leakage.py); LaECE identity from matrix_summary.json cells"),
    "sources": {
        "mAP50": "dfire_leakage.json cells.<run>__to__d_fire_test_dedup.{mAP50_clean,mAP50_full,mAP50_leaky}",
        "val_mAP50": "runs_meta/<run>.json best_val_mAP50 / best_epoch",
        "LaECE": "matrix_summary.json cells.<run>__to__<target>.metrics.{identity,platt_scaling}.LaECE_0, .identity.lrp_fn",
    },
    "runs": {},
}
for run in RUNS:
    meta = json.load(open(A / "runs_meta" / f"{run}.json", encoding="utf-8"))
    lc = leak[f"{run}__to__d_fire_test_dedup"]
    r = {
        "best_epoch": meta.get("best_epoch"),
        "val_mAP50": meta.get("best_val_mAP50"),
        "mAP50_d_fire_test_dedup": lc["mAP50_clean"],
        "mAP50_d_fire_test": lc["mAP50_full"],
        "mAP50_leaky_images": lc["mAP50_leaky"],
    }
    for tgt in TARGETS:
        c = matrix[f"{run}__to__{tgt}"]["metrics"]
        r[f"LaECE_{tgt}"] = c["identity"]["LaECE_0"]
        r[f"platt_{tgt}"] = c["platt_scaling"]["LaECE_0"]
        r[f"FN_{tgt}"] = c["identity"]["lrp_fn"]
    out["runs"][run] = r

json.dump(out, open(A / "dedup_training_comparison.json", "w", encoding="utf-8"), indent=2)
for run, r in out["runs"].items():
    print(f"{run:22s} val {r['val_mAP50']:.4f} clean {r['mAP50_d_fire_test_dedup']:.4f} "
          f"official {r['mAP50_d_fire_test']:.4f} leaky {r['mAP50_leaky_images']:.4f} "
          f"LaECE clean {100*r['LaECE_d_fire_test_dedup']:.1f}")
