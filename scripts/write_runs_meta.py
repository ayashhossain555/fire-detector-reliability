"""Write analysis/runs_meta/<run>.json for every finished training run that
lacks one (same schema as the hand-written 2026-09-01..03 files): model,
dataset, seed, epochs run, best epoch + val mAP (from results.csv, best =
max mAP50-95 as ultralytics' fitness proxy is not stored; we record BOTH the
argmax-mAP50-95 epoch and the epoch of best.pt's own stored metrics), config,
library versions, GPU and the SHA-256 of best.pt (checkpoint identity).
Idempotent. Usage: python analysis/write_runs_meta.py [run_name ...]
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS, META = HERE / "runs", HERE / "runs_meta"
DATASET_NOTE = {
    "d_fire_yolo": "d_fire_yolo (HF badsaarow/d-fire, verified exact vs official; val=seeded 10% of train)",
    "pyro_sdis_yolo": "pyro_sdis_yolo (HF pyronear/pyro-sdis, canonical)",
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def yaml_val(text, key):
    m = re.search(rf"^{key}:\s*(.*)$", text, re.M)
    return m.group(1).strip().strip("'\"") if m else None


def main(names):
    META.mkdir(exist_ok=True)
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or (names and d.name not in names):
            continue
        best, res, args = d / "weights" / "best.pt", d / "results.csv", d / "args.yaml"
        out = META / f"{d.name}.json"
        if out.exists() or not (best.exists() and res.exists() and (d / "results.png").exists()):
            continue
        rows = list(csv.DictReader(res.open()))
        rows = [{k.strip(): v for k, v in r.items()} for r in rows]
        m5095 = [float(r["metrics/mAP50-95(B)"]) for r in rows]
        b = int(max(range(len(rows)), key=lambda i: m5095[i]))
        a = args.read_text(encoding="utf-8")
        ds = Path(yaml_val(a, "data")).parent.name
        try:
            import torch, ultralytics
            versions = {"ultralytics": ultralytics.__version__, "torch": torch.__version__}
            gpu = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
        except Exception:
            versions, gpu = {}, None
        meta = {"run": d.name, "model": yaml_val(a, "model"), "dataset": DATASET_NOTE.get(ds, ds),
                "seed": int(yaml_val(a, "seed")), "epochs_run": len(rows),
                "best_epoch": int(rows[b]["epoch"]),
                "best_val_mAP50": round(float(rows[b]["metrics/mAP50(B)"]), 5),
                "best_val_mAP50_95": round(m5095[b], 5),
                "final_epoch_mAP50": round(float(rows[-1]["metrics/mAP50(B)"]), 5),
                "workers": int(yaml_val(a, "workers")), "batch": int(yaml_val(a, "batch")),
                "imgsz": int(yaml_val(a, "imgsz")), **versions, "gpu": gpu,
                "wall_s_total": round(float(rows[-1]["time"]), 1) if "time" in rows[-1] else None,
                "best_pt_sha256": sha256(best), "written_by": "write_runs_meta.py"}
        out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("[ok]", out.name, f"best ep {meta['best_epoch']} mAP50 {meta['best_val_mAP50']}")


if __name__ == "__main__":
    main(sys.argv[1:])
