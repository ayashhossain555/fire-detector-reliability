"""Export detections in COCO bbox.json format for the calibration pipeline.

Runs a trained ultralytics model over the images listed in a canonical GT
JSON (from make_coco_gt.py) and writes [{image_id, category_id, bbox, score}]
with the GT's integer ids and canonical category ids mapped BY NAME from the
model's own class names (so a model trained with any class order exports
correctly).

Protocol settings (TRAINING-PROTOCOL.md): conf 0.001, max_det 300, imgsz 640,
no TTA. The same export feeds BOTH mAP eval and calibration.

Usage:
  .venv/Scripts/python analysis/export_predictions.py \
      --weights analysis/runs/<run>/weights/best.pt \
      --gt analysis/coco_gt/<tag>.json \
      --out analysis/detections/<model>__<tag>.bbox.json
"""

import argparse
import json
from pathlib import Path

CANON = {"fire": 1, "smoke": 2, "human": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    model = YOLO(args.weights)
    name_of = model.names  # {idx: name}

    dets = []
    paths = [(im["id"], im["abs_path"]) for im in gt["images"]]
    for i in range(0, len(paths), args.batch):
        chunk = paths[i:i + args.batch]
        results = model([p for _, p in chunk], conf=args.conf, max_det=args.max_det,
                        imgsz=args.imgsz, device=args.device, verbose=False)
        for (img_id, _), r in zip(chunk, results):
            for b in r.boxes:
                name = name_of[int(b.cls.item())]
                if name not in CANON:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                dets.append({"image_id": img_id, "category_id": CANON[name],
                             "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                             "score": round(b.conf.item(), 6)})
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(dets), encoding="utf-8")
    meta = {"weights": str(Path(args.weights).resolve()), "gt": args.gt,
            "conf": args.conf, "max_det": args.max_det, "imgsz": args.imgsz,
            "n_images": len(paths), "n_detections": len(dets)}
    outp.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ok] {outp.name}: {len(dets)} detections over {len(paths)} images")


if __name__ == "__main__":
    main()
