"""Build canonical COCO ground-truth JSONs from the YOLO-layout datasets.

Canonical category space (by NAME, shared across all datasets):
  fire -> id 1, smoke -> id 2, human -> id 3 (thesis set only)
Datasets disagree on YOLO class order (D-Fire 0=smoke/1=fire; FASDD mirror
0=fire/1=smoke; Pyro-SDIS 0=smoke) — mapping is read from each data.yaml,
never assumed.

Outputs analysis/coco_gt/<dataset>_<split>.json with integer image ids and
an `abs_path` per image (used by export_predictions.py; harmless extra key
for pycocotools). Also handles the seeded halving of pyro-sdis val into
calval/caltest (no native test split; seed 3407; documented in the output).

Usage: .venv/Scripts/python analysis/make_coco_gt.py <dataset_dir_name> ...
       (e.g. pyro_sdis_yolo d_fire_yolo fasdd_cv_yolo)
"""

import json
import random
import re
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "coco_gt"

CANON = {"fire": 1, "smoke": 2, "human": 3}
SEED = 3407


def read_names(ds_dir: Path) -> list[str]:
    yaml_text = (ds_dir / "data.yaml").read_text(encoding="utf-8")
    m = re.search(r"names:\s*\[(.*?)\]", yaml_text)
    if not m:
        raise ValueError(f"no names in {ds_dir}/data.yaml")
    return [n.strip().strip("'\"") for n in m.group(1).split(",")]


def build(ds_dir: Path, split: str, images: list[Path], names: list[str], tag: str, note: str = ""):
    cats = sorted(({"id": CANON[n], "name": n} for n in names if n in CANON), key=lambda c: c["id"])  # sorted: toolbox aligns thresholds by sorted catId
    imgs, anns = [], []
    ann_id = 1
    for img_id, ip in enumerate(sorted(images), start=1):
        with Image.open(ip) as im:
            w, h = im.size
        imgs.append({"id": img_id, "file_name": ip.name, "width": w, "height": h,
                     "abs_path": str(ip.resolve())})
        lp = ds_dir / split / "labels" / (ip.stem + ".txt")
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").strip().splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                cls, cx, cy, bw, bh = int(float(p[0])), *map(float, p[1:])
                name = names[cls] if cls < len(names) else None
                if name not in CANON:
                    continue
                x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
                anns.append({"id": ann_id, "image_id": img_id, "category_id": CANON[name],
                             "bbox": [round(x, 2), round(y, 2), round(bw * w, 2), round(bh * h, 2)],
                             "area": round(bw * w * bh * h, 2), "iscrowd": 0})
                ann_id += 1
    OUT.mkdir(exist_ok=True)
    out = OUT / f"{tag}.json"
    out.write_text(json.dumps({
        "info": {"description": f"{ds_dir.name} {split} -> canonical fire/smoke COCO GT",
                 "note": note, "source_split": split, "n_images": len(imgs), "n_annotations": len(anns)},
        "images": imgs, "annotations": anns, "categories": cats}), encoding="utf-8")
    print(f"[ok] {out.name}: {len(imgs)} imgs, {len(anns)} anns, cats={[c['name'] for c in cats]}")


def main(ds_name: str):
    ds_dir = DATA / ds_name
    names = read_names(ds_dir)
    splits = [d.name for d in ds_dir.iterdir() if (d / "images").is_dir()]
    base = ds_name.replace("_yolo", "")
    if ds_name == "pyro_sdis_yolo":
        # no native test: halve val (seeded) into calval (fit) / caltest (eval)
        val_imgs = sorted((ds_dir / "val" / "images").glob("*.jpg"))
        random.Random(SEED).shuffle(val_imgs)
        half = len(val_imgs) // 2
        note = f"seeded half of val (seed {SEED}); no native test split"
        build(ds_dir, "val", val_imgs[:half], names, f"{base}_calval", note)
        build(ds_dir, "val", val_imgs[half:], names, f"{base}_caltest", note)
        return
    for split in splits:
        if split == "train":
            continue  # GT for eval splits only
        imgs = sorted((ds_dir / split / "images").glob("*.jpg"))
        tag = f"{base}_{'calval' if split in ('val', 'valid') else split}"
        build(ds_dir, split, imgs, names, tag)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        main(a)
