"""Build a unified per-dataset inventory (phase 1).

For each dataset present under data/, writes analysis/datasets/<name>.json:
  {name, source, licence, format, splits: {split: {n_images, n_boxes_by_class,
   img_size_stats}}, class_names, notes}
Every later stage (audit, training, calibration) reads these instead of
re-walking the raw data. Counts here are the paper's provenance record.

Usage: .venv/Scripts/python analysis/build_dataset_index.py [--dataset NAME]
"""

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "datasets"


def yolo_zip_inventory(zpath: Path, class_names: list[str], img_exts=(".jpg", ".jpeg", ".png")):
    """Inventory a YOLO-format zip (train/valid/test x images/labels)."""
    z = zipfile.ZipFile(zpath)
    names = z.namelist()
    splits = defaultdict(lambda: {"n_images": 0, "n_boxes_by_class": Counter(), "n_empty_labels": 0})
    label_files = [n for n in names if n.endswith(".txt") and "/labels/" in n]
    img_count = Counter()
    for n in names:
        if n.lower().endswith(img_exts) and "/images/" in n:
            img_count[n.split("/")[0]] += 1
    for split, cnt in img_count.items():
        splits[split]["n_images"] = cnt
    for lf in label_files:
        split = lf.split("/")[0]
        raw = z.read(lf).decode("utf-8", errors="replace").strip()
        if not raw:
            splits[split]["n_empty_labels"] += 1
            continue
        for line in raw.splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cls = int(float(parts[0]))
                label = class_names[cls] if cls < len(class_names) else f"cls{cls}"
            except ValueError:
                label = "unparseable"
            splits[split]["n_boxes_by_class"][label] += 1
    return {s: {"n_images": v["n_images"],
                "n_boxes_by_class": dict(v["n_boxes_by_class"]),
                "n_empty_labels": v["n_empty_labels"]}
            for s, v in splits.items()}


def coco_zip_inventory(zpath: Path):
    """Inventory a zip holding <split>/_annotations.coco.json (Roboflow COCO)."""
    z = zipfile.ZipFile(zpath)
    out = {}
    for n in z.namelist():
        if n.endswith("_annotations.coco.json"):
            split = n.split("/")[0]
            d = json.loads(z.read(n))
            cats = {c["id"]: c["name"] for c in d["categories"]}
            boxes = Counter(cats[a["category_id"]] for a in d["annotations"])
            out[split] = {"n_images": len(d["images"]),
                          "n_boxes_by_class": dict(boxes)}
    return out


DATASETS = {
    "fasdd_cv_mirror": {
        "builder": lambda: yolo_zip_inventory(DATA / "FASDD_CV.zip", ["fire", "smoke"]),
        "source": "HF weslleyskah/fire_smoke_dataset_fasdd_cv (third-party Roboflow re-export of FASDD_CV)",
        "licence": "official FASDD: CC BY-SA 4.0 (scidb.cn, read 2026-09-01); mirror card mislabels CC BY 4.0",
        "format": "YOLO zip",
        "notes": "PIPELINE DEVELOPMENT ONLY — resplit ~80/10/10, 188 images short of official 95,314, "
                 "auto-orient applied. Paper numbers require official ScienceDB FASDD_CV.",
    },
    "final_fire_project_v4": {
        "builder": lambda: coco_zip_inventory(DATA / "final-fire-project_v4_coco.zip"),
        "source": "Roboflow brac-university-uuemy/final-fire-project v4 (author's 2023 thesis dataset)",
        "licence": "Public Domain (Roboflow API: public=true, license='Public Domain', read 2026-08-31)",
        "format": "COCO zip",
        "notes": "Held-out shifted eval + label-audit case. Cite thesis hdl.handle.net/10361/20208.",
    },
    # d-fire, pyro-sdis: added once their local copies land (HF snapshot / official).
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="only this dataset key")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for key, spec in DATASETS.items():
        if args.dataset and key != args.dataset:
            continue
        try:
            splits = spec["builder"]()
        except FileNotFoundError as e:
            print(f"[skip] {key}: {e}")
            continue
        rec = {"name": key, "source": spec["source"], "licence": spec["licence"],
               "format": spec["format"], "notes": spec["notes"], "splits": splits}
        path = OUT / f"{key}.json"
        path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        total = sum(v["n_images"] for v in splits.values())
        print(f"[ok] {key}: {total} images across {list(splits)} -> {path.name}")


if __name__ == "__main__":
    main()
