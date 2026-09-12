"""Deployment operating-point analysis for every transfer cell.

The confidence threshold IS the product in an alarm system. For each
(run, source calval, target test) cell this computes, from the SAME exported
detections the calibration battery uses:

  1. Box-level PR at IoU 0.5 (greedy score-ordered matching, per class) on
     the source calval; the F1-optimal threshold t_src (what a practitioner
     would ship).
  2. On the target test split, at t_src: box precision / recall / F1, and the
     target's own F1-optimal threshold t_tgt + its F1 (the ceiling the
     practitioner cannot reach without target labels). Threshold drift =
     t_tgt - t_src.
  3. IMAGE-LEVEL alarm metrics at t_src on the target: positive image = has
     >= 1 GT box of a class the model predicts; alarm = >= 1 detection with
     score >= t. Sensitivity (TPR), false-alarm rate (FPR on negative
     images), alarm precision, and the whole ROC (image-level) so that
     "threshold needed for 95 % sensitivity on target" can be read off.
Writes analysis/operating_points.json (one entry per cell) — every number in
the deployment section traces here.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GT = HERE / "coco_gt"
DETS = HERE / "detections"
CAL = HERE / "calibration"
IOU = 0.5


def load_gt(tag):
    g = json.loads((GT / f"{tag}.json").read_text(encoding="utf-8"))
    boxes = {}
    for a in g["annotations"]:
        boxes.setdefault(a["image_id"], []).append((a["category_id"], a["bbox"]))
    return [im["id"] for im in g["images"]], boxes, {c["id"] for c in g["categories"]}


def load_dets(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    by_img = {}
    for x in d:
        by_img.setdefault(x["image_id"], []).append((x["category_id"], x["bbox"], x["score"]))
    return by_img


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def match(img_ids, gt_boxes, dets_by_img, classes):
    """Greedy per-image, per-class matching at IoU>=0.5 in descending score.
    Returns arrays: det scores, det is_tp; total number of GT boxes; per-image
    (max score, is_positive)."""
    scores, tps, n_gt = [], [], 0
    per_img = []
    for iid in img_ids:
        g = [(c, b) for c, b in gt_boxes.get(iid, []) if c in classes]
        n_gt += len(g)
        d = sorted([x for x in dets_by_img.get(iid, []) if x[0] in classes], key=lambda x: -x[2])
        used = [False] * len(g)
        for c, b, s in d:
            best, bj = 0.0, -1
            for j, (gc, gb) in enumerate(g):
                if used[j] or gc != c:
                    continue
                v = iou_xywh(b, gb)
                if v > best:
                    best, bj = v, j
            tp = best >= IOU
            if tp:
                used[bj] = True
            scores.append(s)
            tps.append(tp)
        per_img.append((max([x[2] for x in d], default=0.0), len(g) > 0))
    return np.array(scores), np.array(tps, dtype=bool), n_gt, per_img


def pr_curve(scores, tps, n_gt):
    order = np.argsort(-scores)
    s, t = scores[order], tps[order]
    ctp = np.cumsum(t)
    cfp = np.cumsum(~t)
    prec = ctp / np.maximum(ctp + cfp, 1)
    rec = ctp / max(n_gt, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return s, prec, rec, f1


def at_threshold(s, prec, rec, f1, t):
    k = np.searchsorted(-s, -t, side="right") - 1  # last index with score >= t
    if k < 0:
        return {"precision": None, "recall": 0.0, "f1": 0.0}
    return {"precision": float(prec[k]), "recall": float(rec[k]), "f1": float(f1[k])}


def image_level(per_img, t):
    pos = np.array([p for _, p in per_img])
    mx = np.array([m for m, _ in per_img])
    alarm = mx >= t
    tpr = float(alarm[pos].mean()) if pos.any() else None
    fpr = float(alarm[~pos].mean()) if (~pos).any() else None
    prec = float(pos[alarm].mean()) if alarm.any() else None
    return {"sensitivity": tpr, "false_alarm_rate": fpr, "alarm_precision": prec,
            "n_pos": int(pos.sum()), "n_neg": int((~pos).sum())}


def image_roc(per_img):
    pos = np.array([p for _, p in per_img])
    mx = np.array([m for m, _ in per_img])
    ts = np.linspace(0.0, 1.0, 101)
    return [{"t": round(float(t), 2), "tpr": float((mx[pos] >= t).mean()) if pos.any() else None,
             "fpr": float((mx[~pos] >= t).mean()) if (~pos).any() else None} for t in ts]


def threshold_for_sensitivity(per_img, target=0.95):
    pos = np.array([p for _, p in per_img])
    mx = np.sort(np.array([m for m, p in per_img if p]))
    if len(mx) == 0:
        return None
    k = int(np.floor((1 - target) * len(mx)))
    return float(mx[k])


def main():
    out = {"iou": IOU, "cells": {}}
    for art in sorted(CAL.glob("*.json")):
        cell = json.loads(art.read_text(encoding="utf-8"))
        inp = cell["inputs"]
        src_tag = Path(inp["calval-gt"]).stem
        tgt_tag = Path(inp["test-gt"]).stem
        run = cell["cell"].split("__to__")[0]
        s_ids, s_gt, s_cls = load_gt(src_tag)
        t_ids, t_gt, t_cls = load_gt(tgt_tag)
        s_dets = load_dets(inp["calval-dets"])
        t_dets = load_dets(inp["test-dets"])
        model_cls = {c for v in s_dets.values() for c, _, _ in v} | {c for v in t_dets.values() for c, _, _ in v}
        cls_s = s_cls & model_cls
        cls_t = t_cls & model_cls  # classes both predictable and annotated on target
        sc, tp, ng, s_img = match(s_ids, s_gt, s_dets, cls_s)
        S = pr_curve(sc, tp, ng)
        t_src = float(S[0][int(np.argmax(S[3]))]) if len(sc) else None
        tc, tt, ngt, t_img = match(t_ids, t_gt, t_dets, cls_t)
        T = pr_curve(tc, tt, ngt)
        t_tgt = float(T[0][int(np.argmax(T[3]))]) if len(tc) else None
        entry = {
            "run": run, "source_calval": src_tag, "target_test": tgt_tag,
            "classes_evaluated": sorted(cls_t),
            "source": {"t_f1opt": t_src, "f1_at_t": float(S[3].max()) if len(sc) else None, "n_gt": int(ng),
                       "image_level_at_t": image_level(s_img, t_src) if t_src is not None else None},
            "target": {
                "n_gt": int(ngt),
                "box_at_t_src": at_threshold(*T, t_src) if (t_src is not None and len(tc)) else None,
                "t_f1opt_target": t_tgt, "f1_at_target_opt": float(T[3].max()) if len(tc) else None,
                "threshold_drift": (t_tgt - t_src) if (t_src is not None and t_tgt is not None) else None,
                "image_level_at_t_src": image_level(t_img, t_src) if t_src is not None else None,
                "t_for_95pct_sensitivity": threshold_for_sensitivity(t_img, 0.95),
                "image_level_at_t95": None,
                "image_roc": image_roc(t_img),
            },
        }
        t95 = entry["target"]["t_for_95pct_sensitivity"]
        if t95 is not None:
            entry["target"]["image_level_at_t95"] = image_level(t_img, t95)
        out["cells"][cell["cell"]] = entry
        b = entry["target"]["box_at_t_src"] or {}
        il = entry["target"]["image_level_at_t_src"] or {}
        print(f"{cell['cell']}: t_src {t_src:.3f} (F1 {entry['source']['f1_at_t']:.3f}) -> target F1 {b.get('f1', float('nan')):.3f} "
              f"(opt {entry['target']['f1_at_target_opt']:.3f} @ {t_tgt:.3f}); img sens {il.get('sensitivity')} FAR {il.get('false_alarm_rate')}")
    (HERE / "operating_points.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("artefact ->", HERE / "operating_points.json")


if __name__ == "__main__":
    main()
