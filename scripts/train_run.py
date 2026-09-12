"""Single training run for Paper 05 (protocol: analysis/TRAINING-PROTOCOL.md).

Reproducible launcher for the ultralytics runs (earlier runs were launched
inline with identical arguments; the args.yaml in each run dir is the record).
Usage: python analysis/train_run.py --model rtdetr-l.pt --data <data.yaml>
       --name rtdetrl_pyrosdis_s3407 [--seed 3407 --batch 16 --workers 2]
       [--resume]   # resume from runs/<name>/weights/last.pt
"""
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="")          # '' = auto → CUDA:0 (5070 Ti)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    from ultralytics import YOLO, RTDETR
    cls = RTDETR if "rtdetr" in a.model.lower() else YOLO

    if a.resume:
        last = RUNS / a.name / "weights" / "last.pt"
        if not last.exists():
            sys.exit(f"--resume requested but {last} missing")
        cls(str(last)).train(resume=True)
        return

    cls(a.model).train(
        data=a.data, epochs=a.epochs, patience=a.patience, batch=a.batch,
        imgsz=a.imgsz, device=a.device, workers=a.workers,
        project=str(RUNS), name=a.name, exist_ok=True, pretrained=True,
        seed=a.seed, deterministic=True,
    )

if __name__ == "__main__":
    main()
