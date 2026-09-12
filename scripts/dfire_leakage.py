"""D-Fire train->test leakage: verify the pHash-exact matches at byte / pixel
level, build a de-duplicated official-test subset, and re-score the existing
D-Fire in-domain cells on it (no inference re-run: detections are filtered
by image id from the existing exports).

Steps
 1. From datasets/phash_index.parquet: every test image whose pHash has an
    EXACT (Hamming 0) match in train -> compare with the matched train file:
    byte-identical (MD5), pixel-identical (same size, max abs diff 0),
    near-identical (resized to common size, mean abs diff <= 2/255) or other.
 2. Clean test = test images with NO train image within Hamming <= 8
    (the audit's loosest radius). coco_gt/d_fire_test_dedup.json (+ smokeonly).
 3. For every run with detections on d_fire_test, write filtered detections
    and run the calibration battery -> calibration/<run>__to__d_fire_test_dedup*.json
    plus mAP50 on leaky vs clean via pycocotools.
Writes analysis/dfire_leakage.json.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "d_fire_yolo"
GT, DETS = HERE / "coco_gt", HERE / "detections"
PY = sys.executable


def popcount(a):
    a = a.astype(np.uint64); c = np.zeros(a.shape, dtype=np.int64)
    for _ in range(64):
        c += (a & np.uint64(1)).astype(np.int64); a >>= np.uint64(1)
    return c


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def pixel_compare(a, b):
    with Image.open(a) as ia, Image.open(b) as ib:
        ia, ib = ia.convert("L"), ib.convert("L")
        same_size = ia.size == ib.size
        if not same_size:
            ib = ib.resize(ia.size)
        d = np.abs(np.asarray(ia, dtype=np.int16) - np.asarray(ib, dtype=np.int16))
    return same_size, int(d.max()), float(d.mean())


def coco_map50(gt_path, dets):
    import contextlib, io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO(str(gt_path)); d = g.loadRes(dets) if dets else None
        if d is None:
            return None
        e = COCOeval(g, d, "bbox"); e.evaluate(); e.accumulate(); e.summarize()
    return float(e.stats[1])


def main():
    df = pd.read_parquet(HERE / "datasets" / "phash_index.parquet")
    df = df[df.dataset == "d_fire"].copy(); df["phash"] = df["phash"].astype(np.uint64)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    tr_hash = np.unique(tr.phash.values)
    tr_file = {int(h): f for h, f in zip(tr.phash.values, tr.file.values)}
    mind = np.full(len(te), 99); arg = np.zeros(len(te), dtype=np.uint64)
    th = te.phash.values
    for i in range(0, len(th), 1000):
        d = popcount(np.bitwise_xor(th[i:i+1000, None], tr_hash[None, :]))
        mind[i:i+1000] = d.min(axis=1); arg[i:i+1000] = tr_hash[d.argmin(axis=1)]
    te = te.assign(min_ham=mind, match=[tr_file[int(h)] for h in arg])
    # 1. verify exact matches
    ver = {"byte_identical": 0, "pixel_identical": 0, "near_identical_mad<=2": 0, "other": 0, "examples_other": []}
    for r in te[te.min_ham == 0].itertuples():
        a, b = DATA / "test" / "images" / r.file, DATA / "train" / "images" / r.match
        if md5(a) == md5(b):
            ver["byte_identical"] += 1; continue
        same, mx, mad = pixel_compare(a, b)
        if same and mx == 0:
            ver["pixel_identical"] += 1
        elif mad <= 2.0:
            ver["near_identical_mad<=2"] += 1
        else:
            ver["other"] += 1
            if len(ver["examples_other"]) < 10:
                ver["examples_other"].append({"test": r.file, "train": r.match, "same_size": same, "max_abs_diff": mx, "mean_abs_diff": round(mad, 2)})
    # 2. de-duplicated test
    clean = set(te[te.min_ham > 8].file)
    out_gt = {}
    for tag in ("d_fire_test", "d_fire_test_smokeonly"):
        g = json.loads((GT / f"{tag}.json").read_text(encoding="utf-8"))
        keep_img = [im for im in g["images"] if im["file_name"] in clean]
        keep_ids = {im["id"] for im in keep_img}
        g2 = {"info": {**g["info"], "description": g["info"]["description"] + " — DE-DUPLICATED: images with any train pHash match within Hamming 8 removed",
                       "n_images": len(keep_img), "n_annotations": sum(a["image_id"] in keep_ids for a in g["annotations"])},
              "images": keep_img, "annotations": [a for a in g["annotations"] if a["image_id"] in keep_ids], "categories": g["categories"]}
        (GT / f"{tag}_dedup.json").write_text(json.dumps(g2), encoding="utf-8")
        out_gt[tag + "_dedup"] = {"n_images": len(keep_img), "n_annotations": g2["info"]["n_annotations"]}
    # 3. re-score existing runs
    cells = {}
    for detp in sorted(DETS.glob("*__d_fire_test.bbox.json")):
        run = detp.name.split("__")[0]
        dets = json.loads(detp.read_text(encoding="utf-8"))
        g_full = json.loads((GT / "d_fire_test.json").read_text(encoding="utf-8"))
        clean_ids = {im["id"] for im in g_full["images"] if im["file_name"] in clean}
        leaky_ids = {im["id"] for im in g_full["images"]} - clean_ids
        for tag in ("d_fire_test_dedup", "d_fire_test_smokeonly_dedup"):
            if "pyrosdis" in run and tag == "d_fire_test_dedup":
                continue
            if "dfire" in run and tag.endswith("smokeonly_dedup"):
                continue
            src = "d_fire_calval_dedup" if "dfiredd" in run else "d_fire_calval" if "dfire" in run else "pyro_sdis_calval"  # dedup-trained runs fit on the dedup calval
            outd = DETS / f"{run}__{tag}.bbox.json"
            if not outd.exists():  # never overwrite a direct export (run_matrix) — the two differ at 1e-4 and cells must match their inputs
                outd.write_text(json.dumps([d for d in dets if d["image_id"] in clean_ids]), encoding="utf-8")
            cell = f"{run}__to__{tag}"
            if not (HERE / "calibration" / f"{cell}.json").exists():
                subprocess.run([PY, str(HERE / "run_calibration.py"), "--calval-gt", str(GT / f"{src}.json"),
                                "--calval-dets", str(DETS / f"{run}__{src}.bbox.json"), "--test-gt", str(GT / f"{tag}.json"),
                                "--test-dets", str(outd), "--cell", cell, "--plots"], check=True, capture_output=True)
            c = json.loads((HERE / "calibration" / f"{cell}.json").read_text(encoding="utf-8"))["results"]
            full = json.loads((HERE / "calibration" / f"{run}__to__{tag.replace('_dedup', '')}.json").read_text(encoding="utf-8"))["results"]
            # leaky-subset GT for the mAP contrast
            g_leaky = {**g_full, "images": [im for im in g_full["images"] if im["id"] in leaky_ids],
                       "annotations": [a for a in g_full["annotations"] if a["image_id"] in leaky_ids]}
            lp = HERE / "tmp_calib_inputs" / f"{run}__d_fire_test_leaky_gt.json"; lp.parent.mkdir(exist_ok=True)
            lp.write_text(json.dumps(g_leaky), encoding="utf-8")
            cells[cell] = {"LaECE_0": {k: c[k]["LaECE_0"] for k in c}, "LaECE_0_full_test": {k: full[k]["LaECE_0"] for k in full},
                           "lrp_identity": c["identity"]["lrp"], "lrp_identity_full_test": full["identity"]["lrp"],
                           "mAP50_clean": coco_map50(GT / f"{tag}.json", [d for d in dets if d["image_id"] in clean_ids]),
                           "mAP50_leaky": coco_map50(lp, [d for d in dets if d["image_id"] in leaky_ids]),
                           "mAP50_full": coco_map50(GT / tag.replace("_dedup", "").__add__(".json"), dets)}
            print(cell, {k: round(v * 100, 1) for k, v in cells[cell]["LaECE_0"].items()}, "full:",
                  {k: round(v * 100, 1) for k, v in cells[cell]["LaECE_0_full_test"].items()},
                  "mAP50 clean/leaky/full", round(cells[cell]["mAP50_clean"], 3), round(cells[cell]["mAP50_leaky"], 3), round(cells[cell]["mAP50_full"], 3), flush=True)
    out = {"n_test": int(len(te)), "n_test_with_train_match": {f"ham<={t}": int((te.min_ham <= t).sum()) for t in (0, 4, 8)},
           "exact_match_verification": ver, "clean_test": {"rule": "no train image within pHash Hamming 8", "n_images": len(clean), **out_gt},
           "cells": cells}
    (HERE / "dfire_leakage.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1))


if __name__ == "__main__":
    main()
