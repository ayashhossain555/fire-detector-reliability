"""Size-matched CONTROL for the de-duplicated D-Fire experiment: a random
8,316-image subsample of the ORIGINAL (leaky) train (random.Random(3407)),
val/test unchanged. Trained under the same protocol as d_fire_dedup_yolo,
it separates "half the data" from "no test-adjacent frames". Hard links.
Writes data/d_fire_sub_yolo + analysis/datasets/d_fire_sub_yolo.json.
"""
import json, os, random, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC, DST = HERE.parent / "data" / "d_fire_yolo", HERE.parent / "data" / "d_fire_sub_yolo"
N = 8316

def link(s, d):
    d.parent.mkdir(parents=True, exist_ok=True)
    if not d.exists():
        try: os.link(s, d)
        except OSError: shutil.copy2(s, d)

def copy_split(split, names):
    for f in names:
        link(SRC / split / "images" / f, DST / split / "images" / f)
        lp = SRC / split / "labels" / (Path(f).stem + ".txt")
        (DST / split / "labels").mkdir(parents=True, exist_ok=True)
        if lp.exists() and not (DST / split / "labels" / lp.name).exists():
            shutil.copy2(lp, DST / split / "labels" / lp.name)

def main():
    files = sorted(p.name for p in (SRC / "train" / "images").iterdir())
    random.Random(3407).shuffle(files)
    keep = sorted(files[:N])
    copy_split("train", keep)
    for split in ("val", "test"):
        copy_split(split, sorted(p.name for p in (SRC / split / "images").iterdir()))
    (DST / "data.yaml").write_text((SRC / "data.yaml").read_text(encoding="utf-8").replace(str(SRC.resolve()), str(DST.resolve())).replace("d_fire_yolo", "d_fire_sub_yolo"), encoding="utf-8")
    counts = {s: len(list((DST / s / "images").iterdir())) for s in ("train", "val", "test")}
    (HERE / "datasets" / "d_fire_sub_yolo.json").write_text(json.dumps({"name": "d_fire_sub_yolo", "layout": str(DST),
        "rule": f"random {N} of the 15,499 original train images (random.Random(3407)); val/test unchanged — size-matched control for d_fire_dedup_yolo",
        "splits": counts}, indent=2), encoding="utf-8")
    print(counts)

if __name__ == "__main__":
    main()
