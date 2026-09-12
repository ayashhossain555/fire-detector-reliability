# Fixed training protocol — written BEFORE any run (2026-09-01)

Changing anything here after runs begin requires a dated amendment note and
re-running affected configs. This is the paper's methods section in embryo.

## Detector families (phase 2)
| Family | Model | Why |
|---|---|---|
| YOLO CNN | ultralytics YOLOv8s | the community default in fire literature |
| YOLO CNN (newer) | ultralytics YOLO11s | current ultralytics generation |
| DETR | ultralytics RT-DETR-l† | transformer head, different confidence mechanism |

† if RT-DETR-l exceeds 16 GB at batch 16, drop to batch 8 before changing
model size; record the change.

## Hyperparameters (identical across datasets, ultralytics defaults unless
stated)
- imgsz 640, epochs 60, batch 16 (auto-reduce rule above), optimizer auto,
  seed set per run, deterministic=True, patience 15 (early stop on val mAP50-95),
  pretrained COCO weights (ultralytics default checkpoints), workers 8,
  cache disabled (disk), amp default.
- AMENDMENT 2026-09-01 (before any completed run): workers 8 → 4 → **2**.
  Reason: WinError 1455 (Windows commit limit exhausted — spawned workers
  each map the CUDA DLLs; machine has 15.8 GB RAM, commit limit ~33 GB with
  ~29 GB baseline). Failed at 8 (run 1 hang) and again at 4 (run 4);
  workers=2 final. No results affected — no run had produced any epoch.
  Fallback if 2 still fails: workers=0. Applies to all runs.
  ADDENDUM 2026-09-03: rtdetrl_pyrosdis died at epoch 7 with workers=2
  (DataLoader workers OOM-killed) and again at epoch ~1 with workers=0
  (cv2 OutOfMemoryError — the training peak itself exceeds the ~31.5 GB
  commit limit). Author freed the machine's other workloads (2026-09-03
  13:30) → run RESTARTED at the STANDARD config (batch=16/workers=2) for
  matrix uniformity; the batch=8/workers=0 fallback stands ready if it
  fails again. rtdetrl_dfire completed at the standard config throughout.
- Seeds: 3407 (primary); 1337 (replicate, only for the headline
  source→target pairs if time allows — record which).
- No test-time augmentation anywhere. No per-dataset hyperparameter tuning —
  the paper measures what a practitioner gets with defaults, stated as such.

## Class harmonisation (training label space)
Two classes everywhere: `fire`, `smoke`.
- D-Fire: native fire/smoke → unchanged.
- FASDD: native fire/smoke → unchanged.
- Pyro-SDIS: smoke only → single-class `smoke` model; its rows in the
  transfer matrix are smoke-only (stated in the paper; fire column = n/a).
- final-fire-project (eval only, never trained on): fire/smoke kept,
  human/nofire boxes dropped for detection eval; full label set retained in
  the audit.

## Eval protocol (phases 2–3)
- ultralytics val on each TARGET test split: mAP50, mAP50-95, per-class.
- Raw predictions exported (JSON, conf threshold 0.001, max_det 300) for the
  calibration pipeline — calibration analysis NEVER re-runs inference with
  different settings than the mAP eval.
- Calibration: fiveai/detection_calibration (D-ECE, LaECE, reliability
  diagrams) on the exported predictions; post-hoc temperature/Platt/isotonic
  fitted on the SOURCE val split (transfer test) and separately on a
  1,000-image TARGET calibration sample (repair test; sampled with seed 3407,
  disjoint from the target test split).
- Operating points: F1-optimal threshold on source-val (the practitioner
  default) — evaluated on target; plus fixed conf 0.25 (ultralytics default).

## Run bookkeeping
Every run gets `analysis/runs_meta/<run_id>.json`: model, dataset, seed,
data hash (zip sha256 or dir file-count+bytes), ultralytics version, torch
version, GPU, wall time, best-epoch metrics, path to weights. Weights stay
under analysis/runs/ (gitignored); metrics JSONs are committed.

## Data-source rule (from PHASE0-VERIFICATION)
Paper numbers only from: official D-Fire (or count-verified mirror), official
ScienceDB FASDD, HF pyro-sdis (canonical source), Roboflow v4 export
(canonical for the author's set). The FASDD_CV Roboflow mirror trains
pipeline-shakedown models only, and nothing trained on it is reported.

- RESULT NOTE 2026-09-03: `rtdetrl_pyrosdis_s3407` (standard config, post-
  pagefile) EARLY-STOPPED at epoch 16: best fitness was epoch 1 (val mAP50
  0.649 / mAP50-95 0.331); epochs 2–16 never exceeded it (mAP50 0.27–0.56).
  This is the protocol outcome (patience 15, defaults) and its cells are
  reported as such. A LABELLED DEVIATION run `rtdetrl_pyrosdis_s3407_p60`
  (patience 60 = no early stopping, everything else identical) is queued to
  show whether RT-DETR-l recovers on Pyro-SDIS; its cells carry the `_p60`
  suffix and are never pooled with the protocol runs.
