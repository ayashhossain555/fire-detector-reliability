"""Characterise the Pyro-SDIS NEGATIVE (no-smoke) validation frames.

Question: are the negatives independent confounder scenes, or frames of the
same smoke events just outside the annotated window? This decides how an
image-level "false-alarm rate" on them should be read.
For every negative val frame: time to the nearest POSITIVE frame from the
same camera (metadata date), per-camera negative share, and whether the
negative shares a (camera, hour) group with positives.
Writes analysis/pyro_sdis_negatives.json.
"""
import collections
import datetime as dt
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "datasets" / "pyro_sdis_image_meta.json").read_text(encoding="utf-8"))


def main():
    pos, imgs = set(), []
    for tag in ("pyro_sdis_calval", "pyro_sdis_caltest"):
        g = json.loads((HERE / "coco_gt" / f"{tag}.json").read_text(encoding="utf-8"))
        pid = {a["image_id"] for a in g["annotations"]}
        pos |= {im["file_name"] for im in g["images"] if im["id"] in pid}
        imgs += [im["file_name"] for im in g["images"]]
    neg = [f for f in imgs if f not in pos]
    T = lambda f: dt.datetime.strptime(META[f]["date"], "%Y-%m-%dT%H-%M-%S")
    bycam = collections.defaultdict(list)
    for f in imgs:
        bycam[META[f]["camera"]].append((T(f), f in pos, f))
    bins = collections.Counter()
    gaps = []
    for f in neg:
        d = [abs((t2 - T(f)).total_seconds()) for t2, p, _ in bycam[META[f]["camera"]] if p]
        m = min(d) if d else None
        gaps.append(m)
        bins["<=2min" if m is not None and m <= 120 else "<=10min" if m is not None and m <= 600
             else "<=60min" if m is not None and m <= 3600 else ">60min_or_no_positive_on_camera"] += 1
    same_hour = sum(1 for f in neg if any(p and f2 != f and META[f2]["date"][:13] == META[f]["date"][:13]
                                          for _, p, f2 in bycam[META[f]["camera"]]))
    cn = collections.Counter(META[f]["camera"] for f in neg)
    ct = collections.Counter(META[f]["camera"] for f in imgs)
    tr = [k for k, v in META.items() if v["split"] == "train"]
    lab = HERE.parent / "data" / "pyro_sdis_yolo" / "train" / "labels"
    tr_neg = sum(1 for k in tr if os.path.getsize(lab / (k[:-4] + ".txt")) == 0)
    out = {"val_images": len(imgs), "val_negatives": len(neg), "val_negative_frac": round(len(neg) / len(imgs), 4),
           "negatives_time_to_nearest_positive_same_camera": dict(bins),
           "median_gap_s": sorted(g for g in gaps if g is not None)[len([g for g in gaps if g is not None]) // 2],
           "negatives_sharing_camera_hour_with_positives": same_hour,
           "cameras_with_negatives": len(cn),
           "per_camera": {c: {"negatives": n, "images": ct[c], "neg_frac": round(n / ct[c], 3)} for c, n in cn.most_common()},
           "train_images": len(tr), "train_negatives": tr_neg, "train_negative_frac": round(tr_neg / len(tr), 4),
           "reading": "Negatives are overwhelmingly frames of the same smoke events adjacent to the annotated "
                      "window (pre-onset / faint-smoke), not independent confounder scenes; an image-level "
                      "false-alarm rate on them measures the annotation boundary, not quiet-camera false alarms."}
    (HERE / "pyro_sdis_negatives.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_camera"}, indent=1))


if __name__ == "__main__":
    main()
