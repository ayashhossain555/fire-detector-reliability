"""Materialise HF parquet image datasets (pyro-sdis, d-fire mirror) as
YOLO-layout directories for ultralytics training.

Writes data/<name>_yolo/{train,val,test}/{images,labels}/ + data.yaml, and a
manifest JSON (counts + per-class boxes) into analysis/datasets/ so the
inventory is machine-verifiable.

Usage: .venv/Scripts/python analysis/parquet_to_yolo.py pyro-sdis|d-fire
"""

import io
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def rows(src_dir: Path):
    for f in sorted(src_dir.glob("data/*.parquet")):
        split = f.name.split("-")[0]
        t = pq.read_table(f)
        cols = t.column_names
        for i in range(t.num_rows):
            yield split, {c: t.column(c)[i].as_py() for c in cols}, cols


def norm_split(s: str) -> str:
    return {"valid": "val", "validation": "val"}.get(s, s)


def main(name: str):
    src = DATA / name
    out = DATA / f"{name.replace('-', '_')}_yolo"
    stats: dict = {}
    class_names: list[str] = []
    n = 0
    for split, r, cols in rows(src):
        split = norm_split(split)
        img = r.get("image")
        if isinstance(img, dict):  # HF image struct {bytes, path}
            img_bytes, img_name = img.get("bytes"), img.get("path") or f"{n:07d}.jpg"
        else:
            img_bytes, img_name = img, f"{n:07d}.jpg"
        if not img_bytes:
            continue
        stem = Path(img_name).stem
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        (out / split / "images" / f"{stem}.jpg").write_bytes(img_bytes)

        st = stats.setdefault(split, {"n_images": 0, "boxes": Counter()})
        st["n_images"] += 1
        lines = []
        # annotation columns vary: pyro-sdis has 'annotations' (yolo string);
        # d-fire mirror exposes 'objects' dict or a label txt column.
        ann = r.get("annotations") if r.get("annotations") is not None else r.get("label")
        if isinstance(ann, str):
            for line in ann.strip().splitlines():
                p = line.split()
                if len(p) == 5:
                    lines.append(line.strip())
                    st["boxes"][int(float(p[0]))] += 1
        elif isinstance(r.get("objects"), dict):
            obj = r["objects"]
            cats, bboxes = obj.get("categories") or obj.get("category"), obj.get("bbox")
            if cats and bboxes:
                for c, b in zip(cats, bboxes):
                    # bbox format assumed normalised cxcywh; verified on first
                    # rows by the caller before trusting output.
                    lines.append(f"{c} " + " ".join(f"{v:.6f}" for v in b))
                    st["boxes"][int(c)] += 1
        (out / split / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n += 1
        if n % 5000 == 0:
            print(f"  {n} images...", flush=True)

    names = {"pyro-sdis": ["smoke"], "d-fire": ["smoke", "fire"]}.get(name)
    yaml = [f"path: {out.resolve()}"]
    for s in stats:
        yaml.append(f"{s}: {s}/images")
    yaml += [f"nc: {len(names)}", f"names: {names}"]
    (out / "data.yaml").write_text("\n".join(yaml), encoding="utf-8")

    manifest = {"name": name, "layout": str(out),
                "splits": {s: {"n_images": v["n_images"],
                               "n_boxes_by_class": {str(k): c for k, c in v["boxes"].items()}}
                           for s, v in stats.items()},
                "class_names": names}
    mpath = HERE / "datasets" / f"{name.replace('-', '_')}_yolo.json"
    mpath.parent.mkdir(exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["splits"], indent=1))
    print("manifest ->", mpath)


if __name__ == "__main__":
    main(sys.argv[1])
