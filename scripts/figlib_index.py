"""Index the HPWREN FIgLib archive (F:/Ayash/FIgLib) for the deployment /
time-to-detection section — a FOURTH domain (US tower cameras, 1-min frames).

Every per-sequence tarball with a .done marker is extracted once to
F:/Ayash/FIgLib/seq/<sequence>/. Frame names carry the offset in seconds
from visible plume appearance: <unix_ts>_<+/-offset>.jpg  →  image-level
label = offset >= 0 (smoke visible) per the FIgLib/SmokeyNet convention.
Bounding boxes exist only for the 9 sequences with a CSV under labels_csv/
(MinX,MinY,MaxX,MaxY,Filename) → those frames get COCO 'smoke' boxes.

Outputs:
  analysis/datasets/figlib_index.json     per-sequence: camera, n frames,
                                          offset range, has_bb, extracted path
  analysis/coco_gt/figlib_all.json        every frame (abs_path, image-level
                                          label in `smoke_visible`, `offset_s`,
                                          `sequence`, `camera`); boxes for the
                                          labelled subset only
  analysis/coco_gt/figlib_bb.json         labelled-subset frames only (proper
                                          detection GT → calibration cells)
Credit: https://www.hpwren.ucsd.edu/ (required by the archive terms).
"""
import csv
import json
import re
import tarfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = Path("F:/Ayash/FIgLib")
TAR, SEQ, CSVDIR = ROOT / "tar", ROOT / "seq", ROOT / "labels_csv"
FRAME_RE = re.compile(r"^(\d+)_([+-]\d+)\.jpg$")


def extract_all():
    SEQ.mkdir(exist_ok=True)
    done = []
    for marker in sorted(TAR.glob("*.tgz.done")):
        tgz = TAR / marker.name[:-5]
        name = tgz.name[:-4]
        dest = SEQ / name
        if not dest.exists():
            try:
                with tarfile.open(tgz) as t:
                    t.extractall(SEQ)
            except Exception as e:  # corrupt / partial
                print("[extract-fail]", name, e); continue
            if not dest.exists():  # 2020-08-31 and 2024 tars nest Data/HPWREN-FIgLib/HPWREN-FIgLib-Data/<seq>
                cands = [p for p in SEQ.rglob(name) if p.is_dir() and p != dest]
                if cands:
                    cands[0].rename(dest)
        done.append(name)
    return done


def frames_of(seq_dir):
    out = []
    for p in seq_dir.rglob("*.jpg"):
        m = FRAME_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), p))
    return sorted(out, key=lambda x: x[1])


def load_bb():
    bb = {}
    for c in CSVDIR.glob("*.csv"):
        with c.open() as f:
            for row in csv.DictReader(f):
                fn = Path(row["Filename"]).name
                seq = Path(row["Filename"]).parent.name
                bb.setdefault((seq, fn), []).append([int(row["MinX"]), int(row["MinY"]),
                                                     int(row["MaxX"]), int(row["MaxY"])])
    return bb


def main():
    names = extract_all()
    bb = load_bb()
    index, images, anns, ann_id = [], [], [], 1
    bb_images, bb_anns = [], []
    img_id = 0
    for name in names:
        fr = frames_of(SEQ / name)
        if not fr:
            print("[no-frames]", name); continue
        cam = name.split("_", 2)[2] if name.count("_") >= 2 else name
        has_bb = any((name, p.name) in bb for _, _, p in fr)
        index.append({"sequence": name, "camera": cam, "date": name[:8], "n_frames": len(fr),
                      "offset_min": fr[0][1], "offset_max": fr[-1][1],
                      "n_pre": sum(1 for _, o, _ in fr if o < 0), "n_post": sum(1 for _, o, _ in fr if o >= 0),
                      "has_bb": has_bb, "path": str(SEQ / name)})
        for ts, off, p in fr:
            img_id += 1
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                continue
            rec = {"id": img_id, "file_name": p.name, "width": w, "height": h, "abs_path": str(p.resolve()),
                   "sequence": name, "camera": cam, "offset_s": off, "smoke_visible": off >= 0}
            images.append(rec)
            boxes = bb.get((name, p.name), [])
            for x1, y1, x2, y2 in boxes:
                a = {"id": ann_id, "image_id": img_id, "category_id": 2,
                     "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0}
                anns.append(a); ann_id += 1
            if has_bb:
                bb_images.append(rec)
    bb_ids = {im["id"] for im in bb_images}
    bb_anns = [a for a in anns if a["image_id"] in bb_ids]
    cats = [{"id": 2, "name": "smoke"}]
    (HERE / "datasets" / "figlib_index.json").write_text(json.dumps(
        {"source": "HPWREN FIgLib (https://www.hpwren.ucsd.edu/FIgLib/), tarballs from cdn.hpwren.ucsd.edu",
         "n_sequences": len(index), "n_frames": len(images), "n_bb_sequences": sum(i["has_bb"] for i in index),
         "sequences": index}, indent=1), encoding="utf-8")
    (HERE / "coco_gt" / "figlib_all.json").write_text(json.dumps(
        {"info": {"description": "FIgLib all frames; image-level label smoke_visible = offset_s >= 0; boxes only for labelled subset",
                  "n_images": len(images), "n_annotations": len(anns)},
         "images": images, "annotations": anns, "categories": cats}), encoding="utf-8")
    (HERE / "coco_gt" / "figlib_bb.json").write_text(json.dumps(
        {"info": {"description": "FIgLib bounding-box-labelled sequences only (HPWREN-BB CSVs)",
                  "n_images": len(bb_images), "n_annotations": len(bb_anns)},
         "images": bb_images, "annotations": bb_anns, "categories": cats}), encoding="utf-8")
    print(f"sequences {len(index)}, frames {len(images)}, bb sequences {sum(i['has_bb'] for i in index)}, "
          f"bb frames {len(bb_images)}, boxes {len(bb_anns)}")


if __name__ == "__main__":
    main()
