"""Materialise the author's thesis dataset (Roboflow final-fire-project v4,
Public Domain) as canonical COCO GT for use as a held-out SHIFTED EVAL
target (never trained on). Classes fire/smoke kept (canonical ids 1/2);
'human' and 'nofire' boxes dropped for detection eval (kept in the audit).
Outputs: data/final_fire_project_v4/{train,valid,test}/ (extracted zip),
         analysis/coco_gt/thesis_calval.json  (Roboflow 'valid' split → calibrator fitting)
         analysis/coco_gt/thesis_test.json    (Roboflow 'test' split)
         analysis/coco_gt/thesis_test_smokeonly.json (smoke boxes only, for
         smoke-only source models — honest FN accounting).
Cite hdl.handle.net/10361/20208 wherever these cells appear.
"""
import json, zipfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
ZIP = DATA / "final-fire-project_v4_coco.zip"
OUT_DIR = DATA / "final_fire_project_v4"
GT = HERE / "coco_gt"
CANON = {"fire": 1, "smoke": 2}

def convert(split, tag, keep=("fire", "smoke")):
    d = json.loads((OUT_DIR / split / "_annotations.coco.json").read_text(encoding="utf-8"))
    name_of = {c["id"]: c["name"] for c in d["categories"]}
    imgs, anns, ann_id = [], [], 1
    id_map = {}
    dropped = {}
    for i, im in enumerate(sorted(d["images"], key=lambda x: x["file_name"]), start=1):
        id_map[im["id"]] = i
        imgs.append({"id": i, "file_name": im["file_name"], "width": im["width"], "height": im["height"],
                     "abs_path": str((OUT_DIR / split / im["file_name"]).resolve())})
    for a in d["annotations"]:
        n = name_of[a["category_id"]]
        if n not in keep or n not in CANON:
            dropped[n] = dropped.get(n, 0) + 1
            continue
        x, y, w, h = a["bbox"]
        anns.append({"id": ann_id, "image_id": id_map[a["image_id"]], "category_id": CANON[n],
                     "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                     "area": round(w * h, 2), "iscrowd": 0})
        ann_id += 1
    cats = [{"id": CANON[n], "name": n} for n in ("fire", "smoke") if n in keep]
    out = {"info": {"description": f"final-fire-project v4 {split} -> canonical fire/smoke COCO GT",
                    "note": "Roboflow export; human/nofire boxes dropped for detection eval",
                    "source_split": split, "n_images": len(imgs), "n_annotations": len(anns),
                    "dropped_boxes_by_class": dropped},
           "images": imgs, "annotations": anns, "categories": cats}
    (GT / f"{tag}.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] {tag}: {len(imgs)} imgs, {len(anns)} anns, dropped {dropped}")

def main():
    if not OUT_DIR.exists():
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(OUT_DIR)
        print("extracted ->", OUT_DIR)
    convert("valid", "thesis_calval")
    convert("test", "thesis_test")
    convert("test", "thesis_test_smokeonly", keep=("smoke",))

if __name__ == "__main__":
    main()
