"""Build a de-duplicated D-Fire training set (data/d_fire_dedup_yolo) so the
leakage finding (dfire_leakage.json) can be tested at TRAINING time.

Rules (pHash from datasets/phash_index.parquet, imagehash 4.3.2):
  1. Drop every train/val image within Hamming <= 8 of ANY official test
     image (test-clean training).
  2. Collapse pHash-identical groups inside train (and inside val) to one
     representative (first by filename).
  3. Test split = the clean test (coco_gt/d_fire_test_dedup.json images),
     unchanged files.
Images are NTFS hard links (no extra disk); labels copied. Writes data.yaml,
coco_gt/d_fire_calval_dedup.json (calibrator fitting split), and
analysis/datasets/d_fire_dedup_yolo.json (counts + rules).
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "data" / "d_fire_yolo"
DST = HERE.parent / "data" / "d_fire_dedup_yolo"
GT = HERE / "coco_gt"


def popcount(a):
    a = a.astype(np.uint64); c = np.zeros(a.shape, dtype=np.int64)
    for _ in range(64):
        c += (a & np.uint64(1)).astype(np.int64); a >>= np.uint64(1)
    return c


def near_test(hashes, test_hashes, thr=8):
    hashes = hashes.astype(np.uint64); th = np.unique(test_hashes.astype(np.uint64))
    out = np.zeros(len(hashes), dtype=bool)
    for i in range(0, len(hashes), 1000):
        out[i:i+1000] = popcount(np.bitwise_xor(hashes[i:i+1000, None], th[None, :])).min(axis=1) <= thr
    return out


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    df = pd.read_parquet(HERE / "datasets" / "phash_index.parquet")
    df = df[df.dataset == "d_fire"].copy(); df["phash"] = df["phash"].astype(np.uint64)
    test = df[df.split == "test"]
    stats = {"rules": ["drop train/val within pHash Hamming 8 of any test image",
                       "collapse pHash-identical groups within train / val to one image",
                       "test = clean test (no train match within Hamming 8)"], "splits": {}}
    for split in ("train", "val"):
        g = df[df.split == split].sort_values("file")
        nt = near_test(g.phash.values, test.phash.values)
        g = g[~nt]
        before_collapse = len(g)
        g = g.drop_duplicates("phash", keep="first")
        stats["splits"][split] = {"original": int((df.split == split).sum()), "dropped_near_test": int(nt.sum()),
                                  "dropped_within_split_duplicates": int(before_collapse - len(g)), "kept": int(len(g))}
        for f in g.file:
            link(SRC / split / "images" / f, DST / split / "images" / f)
            lp = SRC / split / "labels" / (Path(f).stem + ".txt")
            (DST / split / "labels").mkdir(parents=True, exist_ok=True)
            if lp.exists():
                shutil.copy2(lp, DST / split / "labels" / lp.name)
    clean_test = json.loads((GT / "d_fire_test_dedup.json").read_text(encoding="utf-8"))
    for im in clean_test["images"]:
        link(SRC / "test" / "images" / im["file_name"], DST / "test" / "images" / im["file_name"])
        lp = SRC / "test" / "labels" / (Path(im["file_name"]).stem + ".txt")
        (DST / "test" / "labels").mkdir(parents=True, exist_ok=True)
        if lp.exists():
            shutil.copy2(lp, DST / "test" / "labels" / lp.name)
    stats["splits"]["test"] = {"original": int((df.split == "test").sum()), "kept": len(clean_test["images"])}
    yaml = (SRC / "data.yaml").read_text(encoding="utf-8").replace(str(SRC.resolve()), str(DST.resolve())).replace("d_fire_yolo", "d_fire_dedup_yolo")
    (DST / "data.yaml").write_text(yaml, encoding="utf-8")
    # calval GT for the dedup val
    val_gt = json.loads((GT / "d_fire_calval.json").read_text(encoding="utf-8"))
    keep = {p.name for p in (DST / "val" / "images").iterdir()}
    imgs = [dict(im, abs_path=str((DST / "val" / "images" / im["file_name"]).resolve())) for im in val_gt["images"] if im["file_name"] in keep]
    ids = {im["id"] for im in imgs}
    out = {"info": {**val_gt["info"], "description": val_gt["info"]["description"] + " — de-duplicated (make_dfire_dedup.py)",
                    "n_images": len(imgs), "n_annotations": sum(a["image_id"] in ids for a in val_gt["annotations"])},
           "images": imgs, "annotations": [a for a in val_gt["annotations"] if a["image_id"] in ids], "categories": val_gt["categories"]}
    (GT / "d_fire_calval_dedup.json").write_text(json.dumps(out), encoding="utf-8")
    stats["calval_dedup"] = {"n_images": len(imgs), "n_annotations": out["info"]["n_annotations"]}
    (HERE / "datasets" / "d_fire_dedup_yolo.json").write_text(json.dumps({"name": "d_fire_dedup_yolo", "layout": str(DST), **stats}, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
