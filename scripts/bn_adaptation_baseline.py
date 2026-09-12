"""Label-free baseline requested by a reviewer: test-time batch-norm statistics adaptation
(Schneider et al., NeurIPS 2020, "Improving robustness against common corruptions by covariate
shift adaptation") on the convolutional (YOLOv8 / YOLO11) cross-domain cells of the label-free
threshold evaluation (label_free_threshold_refine.json, family == "yolo"; 34 cells = 17 runs x 2
targets). RT-DETR cells are excluded (transformer backbone; BN adaptation is a CNN method).

Protocol per cell (same cell definition, matching, F1, bootstrap and kept-rate rule as
label_free_threshold_refine.py, whose functions are IMPORTED unchanged):
  unlabelled adaptation set : the SAME 100 target-calval images the kept-rate rule uses in the
                              refine artefact's data-size curve (numpy default_rng(0), the draw
                              sequence of label_free_threshold_refine.py replayed: 10 draws at
                              n = 25, 10 at n = 50, then the FIRST draw at n = 100). Reproduction
                              is asserted: the kept-rate threshold recomputed from the unadapted
                              detections on those images must equal threshold_draws[0] at n = 100.
  BN adaptation             : runs/<run>/weights/best.pt loaded UNFUSED (YOLO(path).model; no
                              .fuse()); every torch.nn.BatchNorm2d has its running statistics reset,
                              momentum = None (cumulative average over forward batches) and is put in
                              train mode ALONE (every other module stays in eval); the 100 images are
                              forwarded under torch.no_grad() with the predictor's own preprocessing
                              (per-image LetterBox(640, auto=True, stride=32) as Model.predict uses
                              with its forced batch = 1; BGR->RGB; /255), grouped by letterboxed shape
                              into batches of <= 16 (machine constraint). The BN layers then go back
                              to eval. The predictor fuses Conv+BN at export time, which folds the
                              ADAPTED statistics into the conv weights (exact algebraic identity).
  export                    : adapted detections on the target TEST split and on the FULL target
                              calval, exactly as export_predictions.py (conf 0.001, max_det 300,
                              imgsz 640, default NMS iou 0.7, batch-1 predictor), written to
                              detections_bnadapt/<run>__<tag>.bbox.json in the same format.
  thresholds evaluated on the target test (box F1 at IoU 0.5; image-level sensitivity / FAR):
    U_t_src          unadapted model @ t_src (the shipped threshold; = refine M0_t_src)
    U_kept_rate_100  unadapted model @ kept-rate threshold from the 100 images (= refine n=100 draw 0)
    U_kept_rate_full unadapted model @ kept-rate threshold from the full calval (= refine M1c)
    U_oracle         unadapted model @ labelled F1-optimal threshold on the target calval (= refine ORC)
    A_t_src          BN-adapted model @ t_src                       ("BN-adapt")
    A_kept_rate_100  BN-adapted model @ kept-rate threshold recomputed from the ADAPTED scores of the
                     same 100 images                                ("BN-adapt + kept-rate")
    A_kept_rate_full BN-adapted model @ kept-rate threshold from the adapted full calval (extra)
    A_oracle         BN-adapted model @ labelled F1-optimal threshold refitted on the adapted target
                     calval (adapted upper bound)
  regret            : F1(U_oracle) - F1(method)  [primary; comparable with the refine artefact] and
                      F1(A_oracle) - F1(method). Image-level bootstrap of the target test (B = 1000,
                      seed 0, the same multinomial weights as the refine artefact) gives CIs for every
                      F1, regret and paired difference; group CIs = percentile CI of the mean over
                      cells of the replicate regret (image bootstrap) and a cell-level bootstrap.
  LaECE_0           : toolbox call of label_free_threshold_pilot.laece0 on the adapted detections
                      kept at each threshold (top-100 per image pre-filter as in the refine; no-op
                      for YOLO).
Sanity block (first cell, recorded in the artefact): (i) the unadapted export through this script's
export function reproduces the existing detections/<run>__<target>.bbox.json F1 at t_src to within
0.002; (ii) this script's adaptation-pass preprocessing equals the predictor's preprocess() tensor on
the first three adaptation images (float precision); (iii) the adapted model's F1 at t_src differs from
the unadapted; (iv) CONTROL: re-estimating the BN statistics on 100 SOURCE-calval images (seed-0 draw)
instead of target images and exporting the target test — if the machinery is right this should stay
close to the unadapted model (the training-time statistics are an EMA over augmented training batches,
so a small change is expected), whereas the target-image adaptation is the treatment.
Per-cell time limit (5 min, job stops) applies to the GPU work (adaptation + exports) plus evaluation;
the LaECE_0 toolbox calls are timed separately.
Writes analysis/bn_adaptation_baseline.json (+ .md). GPU (device 0).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES") or "0"   # refine.py defaults it to "" on import
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operating_points as op  # noqa: E402  (read-only reuse)
import label_free_threshold_pilot as pilot  # noqa: E402  (read-only reuse: GT, DETS, laece0, dataset_of)
import label_free_threshold_refine as lr  # noqa: E402  (read-only reuse: match_indexed, solve_threshold, evaluate, ci)
from export_predictions import CANON  # noqa: E402

REFINE = HERE / "label_free_threshold_refine.json"
OUT_DETS = HERE / "detections_bnadapt"
OUT_JSON = HERE / "bn_adaptation_baseline.json"
OUT_MD = HERE / "bn_adaptation_baseline.md"
N_ADAPT = 100
SEED = 0
B = 1000
BATCH_ADAPT = 16
EXPORT = {"conf": 0.001, "max_det": 300, "imgsz": 640, "chunk": 16, "device": "0"}
IMGSZ = 640
STRIDE = 32
CELL_TIME_LIMIT_S = 300
METHODS = ["U_t_src", "U_kept_rate_100", "U_kept_rate_full", "U_oracle", "A_t_src", "A_kept_rate_100", "A_kept_rate_full", "A_oracle"]
LAECE_METHODS = ["A_t_src", "A_kept_rate_100", "A_oracle"]
GROUPS = {"yolo_all": lambda c: True,
          "dfire_to_pyro_sdis": lambda c: c["train_set"].startswith("dfire") and c["target"] == "pyro_sdis",
          "dfire_to_thesis": lambda c: c["train_set"].startswith("dfire") and c["target"] == "thesis",
          "pyrosdis_to_dfire_smoke": lambda c: c["train_set"] == "pyrosdis" and c["target"] == "d_fire",
          "pyrosdis_to_thesis_smoke": lambda c: c["train_set"] == "pyrosdis" and c["target"] == "thesis"}


# ----------------------------------------------------------------------------- data helpers
def gt_images(tag):
    g = json.loads((pilot.GT / f"{tag}.json").read_text(encoding="utf-8"))
    return [(im["id"], im["abs_path"]) for im in g["images"]]


def replay_draw(n_img):
    """The refine artefact's draw sequence (seed 0): 10 draws at n=25, 10 at n=50, then the first at n=100."""
    rng = np.random.default_rng(lr.SEED)
    for n in (25, 50):
        for _ in range(lr.N_DRAWS):
            rng.choice(n_img, n, replace=False)
    return rng.choice(n_img, N_ADAPT, replace=False)


def subset_view(U, pick):
    keep = np.zeros(U["n_img"], dtype=bool); keep[pick] = True
    m = keep[U["img"]]
    return {"score": U["score"][m], "img": U["img"][m], "rank": U["rank"][m], "n_dets_img": U["n_dets_img"], "n_img": int(len(pick))}


def kept_rate_threshold(view, stat_src):
    order = np.argsort(-view["score"], kind="stable")
    return lr.solve_threshold(view, order, "M1c_kept_rate", stat_src)


def oracle_threshold(U):
    c = op.pr_curve(U["score"], U["tp"], U["n_gt"])
    return float(c[0][int(np.argmax(c[3]))]), float(c[3].max())


# ----------------------------------------------------------------------------- model / adaptation / export
def load_unfused(weights):
    import torch
    from ultralytics import YOLO
    y = YOLO(str(weights))
    net = y.model
    n_bn = sum(isinstance(m, torch.nn.BatchNorm2d) for m in net.modules())
    assert not net.is_fused(), "model loaded fused"
    assert n_bn > 0, "no BatchNorm2d modules found"
    return y, net, n_bn


def letterbox_batches(paths, batch):
    """Per-image LetterBox(640, auto=True, stride=32) exactly as the batch-1 predictor does, then grouped
    by letterboxed shape into batches of <= `batch` (BCHW float RGB in [0, 1])."""
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.patches import imread
    lb = LetterBox((IMGSZ, IMGSZ), auto=True, stride=STRIDE)
    groups = {}
    for p in paths:
        im = imread(p)
        assert im is not None, p
        x = lb(image=im)
        groups.setdefault(x.shape, []).append(x)
    out = []
    for shape in sorted(groups):
        ims = groups[shape]
        for i in range(0, len(ims), batch):
            t = torch.from_numpy(np.stack(ims[i:i + batch])).permute(0, 3, 1, 2).flip(1).contiguous().float().div_(255)
            out.append(t)
    return out


def adapt_bn(net, paths, device):
    """Schneider et al. test-time BN adaptation: reset running stats, momentum=None (cumulative
    average), BN layers in train mode only, forward the unlabelled images under no_grad."""
    import torch
    net.to(device).eval()
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    batches = letterbox_batches(paths, BATCH_ADAPT)
    with torch.no_grad():
        for t in batches:
            net(t.to(device))
    for m in bns:
        m.eval()
    n_tracked = {int(m.num_batches_tracked.item()) for m in bns}
    assert n_tracked == {len(batches)}, n_tracked
    assert not any(m.training for m in net.modules())
    return {"n_bn": len(bns), "n_batches": len(batches), "batch_sizes": [int(t.shape[0]) for t in batches],
            "batch_shapes": sorted({tuple(t.shape[2:]) for t in batches}), "n_images": len(paths)}


def export(y, tag, out_path):
    """Same loop as export_predictions.py (chunks of 16 paths; Model.predict forces batch = 1)."""
    name_of = y.names
    paths = gt_images(tag)
    dets = []
    for i in range(0, len(paths), EXPORT["chunk"]):
        chunk = paths[i:i + EXPORT["chunk"]]
        results = y([p for _, p in chunk], conf=EXPORT["conf"], max_det=EXPORT["max_det"], imgsz=EXPORT["imgsz"], device=EXPORT["device"], verbose=False)
        for (img_id, _), r in zip(chunk, results):
            for b in r.boxes:
                name = name_of[int(b.cls.item())]
                if name not in CANON:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                dets.append({"image_id": img_id, "category_id": CANON[name],
                             "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                             "score": round(b.conf.item(), 6)})
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(dets), encoding="utf-8")
        out_path.with_suffix(".meta.json").write_text(json.dumps({"weights": str(y.ckpt_path), "gt": str(pilot.GT / f"{tag}.json"), **{k: EXPORT[k] for k in ("conf", "max_det", "imgsz")},
                                                                  "n_images": len(paths), "n_detections": len(dets), "bn_adapted": True}, indent=2), encoding="utf-8")
    return dets


def dets_by_img(raw):
    by = {}
    for x in raw:
        by.setdefault(x["image_id"], []).append((x["category_id"], x["bbox"], x["score"]))
    return by


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter on cell name")
    ap.add_argument("--sanity-only", action="store_true", help="run the sanity block on the first cell and stop")
    ap.add_argument("--no-laece", action="store_true")
    ap.add_argument("--B", type=int, default=B)
    args = ap.parse_args()
    import torch
    t_start = time.time()
    rj = json.loads(REFINE.read_text(encoding="utf-8"))
    cells = [(n, c) for n, c in rj["cells"].items() if c["family"] == "yolo"]
    if args.only:
        cells = [(n, c) for n, c in cells if args.only in n]
    print(f"{len(cells)} YOLO cells from {REFINE.name}; device {torch.cuda.get_device_name(0)}", flush=True)
    device = torch.device("cuda:0")

    gt_cache, W_cache = {}, {}

    def gt(tag):
        if tag not in gt_cache:
            gt_cache[tag] = op.load_gt(tag)
        return gt_cache[tag]

    def weights(tag, n_img):
        if tag not in W_cache:
            rng = np.random.default_rng(SEED)
            W_cache[tag] = rng.multinomial(n_img, np.full(n_img, 1.0 / n_img), size=args.B).astype(float)
        assert W_cache[tag].shape == (args.B, n_img)
        return W_cache[tag]

    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "protocol": __doc__, "refine_artefact": str(REFINE), "refine_created": rj["created"],
           "config": {"n_adapt_images": N_ADAPT, "draw": "numpy default_rng(0); refine draw sequence replayed (10x n=25, 10x n=50, first n=100)",
                      "bn": {"momentum": None, "reset_running_stats": True, "train_mode": "BatchNorm2d only", "grad": "torch.no_grad", "batch_max": BATCH_ADAPT,
                             "preprocess": f"per-image LetterBox({IMGSZ}, auto=True, stride={STRIDE}), BGR->RGB, /255; batches grouped by letterboxed shape"},
                      "export": EXPORT, "nms_iou": 0.7, "bootstrap": {"B": args.B, "seed": SEED, "unit": "target-test image (multinomial weights, identical to the refine artefact)"},
                      "laece_prefilter_top_per_image": lr.LAECE_TOP_PER_IMAGE, "ultralytics": __import__("ultralytics").__version__, "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(0), "cell_time_limit_s": CELL_TIME_LIMIT_S},
           "methods": METHODS, "sanity": None, "cells": {}, "problems": []}
    reps = {}
    rows = []
    for k, (name, c) in enumerate(cells):
        t0 = time.time()
        run, src, cal, tgt = c["run"], c["source_calval"], c["target_calval_unlabelled"], c["target_test"]
        cls_t = set(c["classes_evaluated"])
        t_src = c["source"]["t_src"]
        stat_src = c["source"]["statistics_at_t_src"]["M1c_kept_rate"]
        u_ids, u_gt, _ = gt(cal)
        x_ids, x_gt, _ = gt(tgt)
        # --- unadapted side (existing detections; must reproduce the refine artefact) --------------------
        u_d = op.load_dets(c["files"]["target_calval"]); x_d = op.load_dets(c["files"]["target_test"])
        U = lr.match_indexed(u_ids, u_gt, u_d, cls_t); X = lr.match_indexed(x_ids, x_gt, x_d, cls_t)
        pick = replay_draw(U["n_img"])
        adapt_ids = [int(u_ids[i]) for i in pick]
        img_meta = dict(gt_images(cal))
        adapt_paths = [img_meta[i] for i in adapt_ids]
        t_kr_u100 = kept_rate_threshold(subset_view(U, pick), stat_src)
        ref_sub = c["unlabelled_data_sensitivity"]["100"]["M1c_kept_rate"]
        assert t_kr_u100 == ref_sub["threshold_draws"][0], (name, t_kr_u100, ref_sub["threshold_draws"][0])
        t_kr_ufull = kept_rate_threshold(lr.unlabelled_view(U), stat_src)
        assert abs(t_kr_ufull - c["methods"]["M1c_kept_rate"]["threshold"]) < 1e-12
        t_orc_u, f1_orc_u_cal = oracle_threshold(U)
        assert abs(t_orc_u - c["methods"]["ORC_target_calval"]["threshold"]) < 1e-12
        W = weights(tgt, X["n_img"])
        # --- BN adaptation ----------------------------------------------------------------------------
        weights_path = HERE / "runs" / run / "weights" / "best.pt"
        y, net, n_bn = load_unfused(weights_path)
        sanity = None
        if k == 0 or args.sanity_only:
            # (i) unadapted export through this script reproduces the existing target-test detections
            t_s = time.time()
            d_repro = export(y, tgt, None)
            Xr = lr.match_indexed(x_ids, x_gt, dets_by_img(d_repro), cls_t)
            f1_repro = lr.evaluate(Xr, t_src, W)[0]["f1"]
            f1_exist = lr.evaluate(X, t_src, W)[0]["f1"]
            # (ii) preprocessing identity vs the predictor's own preprocess() on the first 3 adaptation images
            from ultralytics.utils.patches import imread
            max_pre_diff = 0.0
            for p in adapt_paths[:3]:
                mine = letterbox_batches([p], 1)[0]
                theirs = y.predictor.preprocess([imread(p)]).float().cpu()
                assert mine.shape == theirs.shape, (mine.shape, theirs.shape)
                max_pre_diff = max(max_pre_diff, float((mine - theirs).abs().max()))
            # (iv) control: BN statistics re-estimated on 100 SOURCE calval images (seed-0 draw), target test exported
            s_ids, _, _ = gt(src)
            src_pick = np.random.default_rng(SEED).choice(len(s_ids), N_ADAPT, replace=False)
            src_meta = dict(gt_images(src))
            src_paths = [src_meta[int(s_ids[i])] for i in src_pick]
            y_c, net_c, _ = load_unfused(weights_path)
            ctrl_info = adapt_bn(net_c, src_paths, device)
            d_ctrl = export(y_c, tgt, None)
            Xctrl = lr.match_indexed(x_ids, x_gt, dets_by_img(d_ctrl), cls_t)
            f1_ctrl = lr.evaluate(Xctrl, t_src, W)[0]["f1"]
            cc = op.pr_curve(Xctrl["score"], Xctrl["tp"], Xctrl["n_gt"])
            del y_c, net_c
            sanity = {"cell": name, "f1_at_t_src_existing_file": f1_exist, "f1_at_t_src_reexported_unadapted": f1_repro,
                      "abs_diff": abs(f1_repro - f1_exist), "passed_0.002": abs(f1_repro - f1_exist) <= 0.002,
                      "n_dets_existing": int(sum(len(v) for v in x_d.values())), "n_dets_reexported": len(d_repro),
                      "preprocess_max_abs_diff_vs_predictor": max_pre_diff, "preprocess_identical": max_pre_diff < 1e-6,
                      "model_fused_by_predictor_after_export": bool(y.model.is_fused()),
                      "source_control": {"what": "BN statistics re-estimated on 100 SOURCE-calval images (same procedure), target test exported",
                                         "source_calval": src, "image_ids": [int(s_ids[i]) for i in src_pick], "adaptation": ctrl_info,
                                         "f1_at_t_src": f1_ctrl, "f1_at_t_src_unadapted": f1_exist, "abs_diff_vs_unadapted": abs(f1_ctrl - f1_exist),
                                         "f1_test_optimal": float(cc[3].max()), "f1_test_optimal_unadapted": float(Xc_s[3].max()) if (Xc_s := op.pr_curve(X["score"], X["tp"], X["n_gt"])) is not None else None,
                                         "n_dets": len(d_ctrl)},
                      "seconds": round(time.time() - t_s, 1)}
            print(f"[sanity] source-control (BN stats from 100 SOURCE images): F1@t_src {f1_ctrl:.4f} vs unadapted {f1_exist:.4f}; "
                  f"test-opt F1 {float(cc[3].max()):.4f} vs {sanity['source_control']['f1_test_optimal_unadapted']:.4f}; dets {len(d_ctrl)}", flush=True)
            print(f"[sanity] {name}: F1@t_src existing {f1_exist:.4f} re-exported {f1_repro:.4f} (|d| = {abs(f1_repro - f1_exist):.4f}); "
                  f"preprocess max|d| = {max_pre_diff:.2e}; dets {sanity['n_dets_existing']} vs {len(d_repro)}", flush=True)
            if not sanity["passed_0.002"]:
                out["problems"].append(f"sanity: unadapted re-export differs from existing file by {abs(f1_repro - f1_exist):.4f} F1 on {name}")
                print("STOP: sanity (i) failed", flush=True)
                break
            y, net, n_bn = load_unfused(weights_path)      # the predictor fused the first copy; reload fresh
        t_a = time.time()
        adapt_info = adapt_bn(net, adapt_paths, device)
        adapt_info["seconds"] = round(time.time() - t_a, 1)
        t_e = time.time()
        d_test = export(y, tgt, OUT_DETS / f"{run}__{tgt}.bbox.json")
        d_cal = export(y, cal, OUT_DETS / f"{run}__{cal}.bbox.json")
        export_s = round(time.time() - t_e, 1)
        Xa = lr.match_indexed(x_ids, x_gt, dets_by_img(d_test), cls_t)
        Ua = lr.match_indexed(u_ids, u_gt, dets_by_img(d_cal), cls_t)
        # --- thresholds -------------------------------------------------------------------------------
        t_kr_a100 = kept_rate_threshold(subset_view(Ua, pick), stat_src)
        t_kr_afull = kept_rate_threshold(lr.unlabelled_view(Ua), stat_src)
        t_orc_a, f1_orc_a_cal = oracle_threshold(Ua)
        thr = {"U_t_src": (X, t_src), "U_kept_rate_100": (X, t_kr_u100), "U_kept_rate_full": (X, t_kr_ufull), "U_oracle": (X, t_orc_u),
               "A_t_src": (Xa, t_src), "A_kept_rate_100": (Xa, t_kr_a100), "A_kept_rate_full": (Xa, t_kr_afull), "A_oracle": (Xa, t_orc_a)}
        res, rep = {}, {}
        for m, (XX, t) in thr.items():
            point, r = lr.evaluate(XX, t, W)
            res[m] = {"threshold": t, "adapted": m.startswith("A_"), **point}
            rep[m] = r
        # exact reproduction of the refine artefact on the unadapted side
        rm = c["methods"]
        for m, rk in (("U_t_src", "M0_t_src"), ("U_kept_rate_full", "M1c_kept_rate"), ("U_oracle", "ORC_target_calval")):
            assert abs(res[m]["f1"] - rm[rk]["f1"]) < 1e-9, (name, m, res[m]["f1"], rm[rk]["f1"])
            assert abs(res[m]["ci95"]["f1"][0] - rm[rk]["ci95"]["f1"][0]) < 1e-9 if "ci95" in res[m] else True
        assert abs((res["U_oracle"]["f1"] - res["U_kept_rate_100"]["f1"]) - ref_sub["regret_draws"][0]) < 1e-9, name
        Xc, Xac = op.pr_curve(X["score"], X["tp"], X["n_gt"]), op.pr_curve(Xa["score"], Xa["tp"], Xa["n_gt"])
        f1_orc_u, f1_orc_a = res["U_oracle"]["f1"], res["A_oracle"]["f1"]
        for m in METHODS:
            r = res[m]
            r["regret_vs_U_oracle"] = f1_orc_u - r["f1"]
            r["regret_vs_A_oracle"] = f1_orc_a - r["f1"]
            r["ci95"] = {q: lr.ci(rep[m][q]) for q in ("f1", "sensitivity", "false_alarm_rate")}
            r["regret_vs_U_oracle_ci95"] = lr.ci(rep["U_oracle"]["f1"] - rep[m]["f1"])
            r["regret_vs_A_oracle_ci95"] = lr.ci(rep["A_oracle"]["f1"] - rep[m]["f1"])
            rep[m]["regret_U"] = rep["U_oracle"]["f1"] - rep[m]["f1"]
            rep[m]["regret_A"] = rep["A_oracle"]["f1"] - rep[m]["f1"]
            for ref in ("U_t_src", "U_kept_rate_100", "U_kept_rate_full"):
                if m == ref:
                    continue
                d = rep[m]["f1"] - rep[ref]["f1"]
                rep[m][f"diff_vs_{ref}"] = d
                r[f"diff_vs_{ref}"] = {"f1": r["f1"] - res[ref]["f1"], "ci95": lr.ci(d), "p_boot_le_0": float(np.mean(d <= 0)),
                                       "sensitivity": (r["sensitivity"] - res[ref]["sensitivity"]) if (r["sensitivity"] is not None and res[ref]["sensitivity"] is not None) else None,
                                       "false_alarm_rate": (r["false_alarm_rate"] - res[ref]["false_alarm_rate"]) if (r["false_alarm_rate"] is not None and res[ref]["false_alarm_rate"] is not None) else None}
        # --- LaECE_0 on the adapted detections -------------------------------------------------------------
        laece_note = None
        secs_gpu_eval = round(time.time() - t0 - (sanity["seconds"] if sanity else 0.0), 1)
        t_l = time.time()
        if not args.no_laece:
            raw = [d for d in d_test if d["category_id"] in cls_t]
            raw_f = lr.top_per_image(raw, lr.LAECE_TOP_PER_IMAGE)
            laece_note = {"n_dets": len(raw), "n_removed_by_prefilter": len(raw) - len(raw_f)}
            for m in LAECE_METHODS:
                kept = [d for d in raw_f if d["score"] >= thr[m][1]]
                res[m]["LaECE_0"] = pilot._q(pilot.laece0, str(pilot.GT / f"{tgt}.json"), kept)
                res[m]["LaECE_0_n_kept"] = len(kept)
            for m, rk in (("U_t_src", "M0_t_src"), ("U_kept_rate_full", "M1c_kept_rate"), ("U_oracle", "ORC_target_calval")):
                res[m]["LaECE_0"] = rm[rk].get("LaECE_0"); res[m]["LaECE_0_n_kept"] = rm[rk].get("LaECE_0_n_kept")
        secs = round(time.time() - t0, 1)
        secs_laece = round(time.time() - t_l, 1)
        if sanity is not None:
            out["sanity"] = {**sanity, "f1_at_t_src_adapted": res["A_t_src"]["f1"], "adapted_differs": abs(res["A_t_src"]["f1"] - res["U_t_src"]["f1"]) > 1e-6,
                             "n_dets_adapted_test": len(d_test), "n_bn": n_bn}
            print(f"[sanity] adapted F1@t_src {res['A_t_src']['f1']:.4f} vs unadapted {res['U_t_src']['f1']:.4f}; adapted dets {len(d_test)}", flush=True)
        cell = {"run": run, "model": c["model"], "train_set": c["train_set"], "seed": c["seed"], "target": c["target"], "target_test": tgt,
                "source_calval": src, "target_calval_unlabelled": cal, "classes_evaluated": sorted(cls_t),
                "weights": str(weights_path), "n_bn_modules": n_bn, "adaptation": adapt_info,
                "adaptation_image_ids": adapt_ids, "adaptation_image_files": [Path(p).name for p in adapt_paths],
                "files": {"target_test_adapted": str(OUT_DETS / f"{run}__{tgt}.bbox.json"), "target_calval_adapted": str(OUT_DETS / f"{run}__{cal}.bbox.json"), **c["files"]},
                "n": {**c["n"], "target_test_dets_adapted": int(len(Xa["score"])), "target_calval_dets_adapted": int(len(Ua["score"]))},
                "source": {"t_src": t_src, "kept_rate_statistic_at_t_src": stat_src},
                "thresholds": {"t_src": t_src, "kept_rate_100_unadapted": t_kr_u100, "kept_rate_full_unadapted": t_kr_ufull, "oracle_unadapted": t_orc_u,
                               "kept_rate_100_adapted": t_kr_a100, "kept_rate_full_adapted": t_kr_afull, "oracle_adapted": t_orc_a},
                "calval_oracle_f1": {"unadapted": f1_orc_u_cal, "adapted": f1_orc_a_cal},
                "target_test_summary": {"f1_test_optimal_unadapted": float(Xc[3].max()), "f1_test_optimal_adapted": float(Xac[3].max()),
                                        "t_test_optimal_adapted": float(Xac[0][int(np.argmax(Xac[3]))]),
                                        "ap_proxy_unadapted_vs_adapted_note": "F1 at the test-optimal threshold: the ceiling for each model"},
                "refine_reference": {"M0_f1": rm["M0_t_src"]["f1"], "M1c_full_f1": rm["M1c_kept_rate"]["f1"], "ORC_f1": rm["ORC_target_calval"]["f1"],
                                     "M1c_n100_draw0_regret": ref_sub["regret_draws"][0], "M1c_n100_draw0_threshold": ref_sub["threshold_draws"][0]},
                "laece": laece_note, "methods": res, "seconds": secs, "seconds_export": export_s, "seconds_laece": secs_laece,
                "seconds_adapt_export_eval_excl_sanity": secs_gpu_eval}
        out["cells"][name] = cell
        reps[name] = rep
        rows.append(name)
        print(f"[{k + 1}/{len(cells)}] {name}: BN {n_bn} ({adapt_info['n_batches']} batches, {adapt_info['seconds']}s) export {export_s}s | "
              f"F1 t_src {res['U_t_src']['f1']:.3f} kr100 {res['U_kept_rate_100']['f1']:.3f} orcU {f1_orc_u:.3f} | "
              f"BN {res['A_t_src']['f1']:.3f} BN+kr100 {res['A_kept_rate_100']['f1']:.3f} orcA {f1_orc_a:.3f} | {secs}s (laece {secs_laece}s)", flush=True)
        if args.sanity_only:
            print("sanity-only run: stopping after the first cell", flush=True)
            break
        if secs_gpu_eval > CELL_TIME_LIMIT_S:
            out["problems"].append(f"{name} took {secs_gpu_eval}s (adapt + export + eval, excl. sanity/LaECE) > {CELL_TIME_LIMIT_S}s; job stopped after this cell as instructed")
            print("STOP: cell exceeded the time limit", flush=True)
            break

    # ---------------------------------------------------------------------- summaries
    C = out["cells"]

    def group_summary(sel, rng_cells):
        s = {}
        for m in METHODS:
            f1 = np.array([C[c]["methods"][m]["f1"] for c in sel])
            e = {"n_cells": len(sel), "f1_mean": float(f1.mean())}
            for key, rk in (("U", "regret_vs_U_oracle"), ("A", "regret_vs_A_oracle")):
                reg = np.array([C[c]["methods"][m][rk] for c in sel])
                reg_rep = np.mean([reps[c][m][f"regret_{key}"] for c in sel], axis=0)
                cb = [reg[rng_cells.integers(0, len(sel), len(sel))].mean() for _ in range(args.B)] if len(sel) >= 2 else None
                e[f"regret_vs_{key}_oracle"] = {"mean": float(reg.mean()), "median": float(np.median(reg)), "max": float(reg.max()),
                                                "sd_across_cells": float(reg.std(ddof=1)) if len(sel) > 1 else None,
                                                "mean_ci95_image_bootstrap": lr.ci(reg_rep), "mean_ci95_cell_bootstrap": lr.ci(cb) if cb is not None else None}
            for ref in ("U_t_src", "U_kept_rate_100", "U_kept_rate_full"):
                if m == ref:
                    continue
                d = np.array([C[c]["methods"][m][f"diff_vs_{ref}"]["f1"] for c in sel])
                d_rep = np.mean([reps[c][m][f"diff_vs_{ref}"] for c in sel], axis=0)
                e[f"vs_{ref}"] = {"f1_gain_mean": float(d.mean()), "f1_gain_ci95_image_bootstrap": lr.ci(d_rep),
                                  "wins": int(np.sum(d > 1e-12)), "losses": int(np.sum(d < -1e-12)), "ties": int(np.sum(np.abs(d) <= 1e-12)),
                                  "wins_ci_excludes_0": int(sum(1 for c in sel if C[c]["methods"][m][f"diff_vs_{ref}"]["ci95"][0] > 0)),
                                  "losses_ci_excludes_0": int(sum(1 for c in sel if C[c]["methods"][m][f"diff_vs_{ref}"]["ci95"][1] < 0))}
            for q in ("sensitivity", "false_alarm_rate"):
                pts = np.array([v for v in (C[c]["methods"][m][q] for c in sel) if v is not None], dtype=float)
                e[f"{q}_mean"] = float(pts.mean()) if len(pts) else None
                e[f"{q}_mean_ci95_image_bootstrap"] = lr.ci(np.nanmean([reps[c][m][q] for c in sel], axis=0))
            la = np.array([v for v in (C[c]["methods"][m].get("LaECE_0") for c in sel) if v is not None], dtype=float)
            e["LaECE_0_mean"] = float(la.mean()) if len(la) else None
            e["LaECE_0_median"] = float(np.median(la)) if len(la) else None
            s[m] = e
        return s

    rng_cells = np.random.default_rng(SEED)
    groups = {g: [c for c in rows if f(C[c])] for g, f in GROUPS.items()}
    summary = {g: group_summary(sel, rng_cells) for g, sel in groups.items() if sel}
    out["summary"] = {"groups": summary, "group_cells": groups, "n_cells": len(rows), "seconds_total": round(time.time() - t_start, 1),
                      "adaptation_seconds_mean": float(np.mean([C[c]["adaptation"]["seconds"] for c in rows])) if rows else None,
                      "cell_seconds_max": max((C[c]["seconds"] for c in rows), default=None),
                      "cell_seconds_adapt_export_eval_max": max((C[c]["seconds_adapt_export_eval_excl_sanity"] for c in rows), default=None),
                      "cell_seconds_laece_max": max((C[c]["seconds_laece"] for c in rows), default=None)}
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------------- md
    def f3(x):
        return "" if x is None else f"{x:.3f}"

    def cis(c):
        return "" if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]"

    sn = out["sanity"] or {}
    L = ["# Test-time BN adaptation baseline (Schneider et al. 2020) vs label-free kept-rate — machine-written by bn_adaptation_baseline.py", "",
         f"Cells: {len(rows)} YOLO (v8 / 11) cross-domain cells from {REFINE.name}. Adaptation set: the same {N_ADAPT} unlabelled target-calval images "
         f"(seed-0 draw) the kept-rate rule uses at n = 100 in the refine artefact (reproduction asserted per cell). Regret = F1(oracle) - F1(method), box F1 at IoU 0.5 "
         f"on the target test; U-oracle = unadapted labelled oracle (as in the refine artefact), A-oracle = labelled oracle refitted on the adapted calval. "
         f"CIs: image-level bootstrap of the target test (B = {args.B}, seed {SEED}); cell-level bootstrap also given. "
         f"Total {out['summary']['seconds_total']} s; max cell {out['summary']['cell_seconds_max']} s.", "",
         f"Sanity ({sn.get('cell')}): unadapted re-export F1@t_src {f3(sn.get('f1_at_t_src_reexported_unadapted'))} vs existing file {f3(sn.get('f1_at_t_src_existing_file'))} "
         f"(|d| = {sn.get('abs_diff', float('nan')):.4f}, pass = {sn.get('passed_0.002')}); preprocessing identical to predictor = {sn.get('preprocess_identical')} "
         f"(max |d| = {sn.get('preprocess_max_abs_diff_vs_predictor')}); adapted F1@t_src {f3(sn.get('f1_at_t_src_adapted'))} (differs = {sn.get('adapted_differs')}); "
         f"BatchNorm2d modules in the first model: {sn.get('n_bn')}. Source-control (BN statistics from 100 SOURCE images): F1@t_src "
         f"{f3((sn.get('source_control') or {}).get('f1_at_t_src'))} vs unadapted {f3(sn.get('f1_at_t_src_existing_file'))}; test-optimal F1 "
         f"{f3((sn.get('source_control') or {}).get('f1_test_optimal'))} vs {f3((sn.get('source_control') or {}).get('f1_test_optimal_unadapted'))}.", ""]
    if out["problems"]:
        L += ["**Problems:** " + "; ".join(out["problems"]), ""]
    L += ["Methods: U_t_src = shipped threshold; U_kept_rate_100 = kept-rate rule on the 100 images (unadapted); U_kept_rate_full = kept-rate on the full calval (refine M1c); "
          "A_t_src = BN-adapt at the shipped threshold; A_kept_rate_100 = BN-adapt + kept-rate on the adapted scores of the same 100 images; A_kept_rate_full = BN-adapt + kept-rate on the full adapted calval; "
          "U_oracle / A_oracle = labelled upper bounds.", ""]
    for g, s in summary.items():
        L += [f"## {g} (n = {s['U_t_src']['n_cells']})", "",
              "| method | F1 mean | regret vs U-oracle mean | CI (image boot) | CI (cell boot) | median | regret vs A-oracle mean | CI (image boot) | gain vs t_src | CI | W/L vs t_src | gain vs kept-rate-100 | CI | W/L vs kept-rate-100 (CI-sig W/L) | gain vs kept-rate-full | W/L | sens | FAR | LaECE_0 median |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, v in s.items():
            rU, rA = v["regret_vs_U_oracle"], v["regret_vs_A_oracle"]
            vt, vk, vf = v.get("vs_U_t_src"), v.get("vs_U_kept_rate_100"), v.get("vs_U_kept_rate_full")
            L.append(f"| {m} | {f3(v['f1_mean'])} | {f3(rU['mean'])} | {cis(rU['mean_ci95_image_bootstrap'])} | {cis(rU['mean_ci95_cell_bootstrap'])} | {f3(rU['median'])} | "
                     f"{f3(rA['mean'])} | {cis(rA['mean_ci95_image_bootstrap'])} | "
                     f"{f3(vt['f1_gain_mean']) if vt else ''} | {cis(vt['f1_gain_ci95_image_bootstrap']) if vt else ''} | {f'{vt['wins']}/{vt['losses']}' if vt else ''} | "
                     f"{f3(vk['f1_gain_mean']) if vk else ''} | {cis(vk['f1_gain_ci95_image_bootstrap']) if vk else ''} | {f'{vk['wins']}/{vk['losses']} ({vk['wins_ci_excludes_0']}/{vk['losses_ci_excludes_0']})' if vk else ''} | "
                     f"{f3(vf['f1_gain_mean']) if vf else ''} | {f'{vf['wins']}/{vf['losses']}' if vf else ''} | {f3(v['sensitivity_mean'])} | {f3(v['false_alarm_rate_mean'])} | {f3(v['LaECE_0_median'])} |")
        L.append("")
    L += ["## Per cell (F1 on the target test [95 % CI]; regret vs U-oracle)", "",
          "| cell | t_src | t_kr100 | t_kr100 adapted | t_orc U | t_orc A | F1 t_src | F1 kr100 | F1 BN-adapt | F1 BN+kr100 | F1 U-oracle | F1 A-oracle | F1 test-opt U / A | BN-adapt - kr100 [CI] | BN+kr100 - kr100 [CI] | regret BN / BN+kr / kr100 | LaECE BN / BN+kr / A-orc | s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cn in rows:
        r, th, ts = C[cn]["methods"], C[cn]["thresholds"], C[cn]["target_test_summary"]
        L.append(f"| {cn} | {f3(th['t_src'])} | {f3(th['kept_rate_100_unadapted'])} | {f3(th['kept_rate_100_adapted'])} | {f3(th['oracle_unadapted'])} | {f3(th['oracle_adapted'])} | "
                 f"{f3(r['U_t_src']['f1'])} {cis(r['U_t_src']['ci95']['f1'])} | {f3(r['U_kept_rate_100']['f1'])} {cis(r['U_kept_rate_100']['ci95']['f1'])} | "
                 f"{f3(r['A_t_src']['f1'])} {cis(r['A_t_src']['ci95']['f1'])} | {f3(r['A_kept_rate_100']['f1'])} {cis(r['A_kept_rate_100']['ci95']['f1'])} | "
                 f"{f3(r['U_oracle']['f1'])} | {f3(r['A_oracle']['f1'])} | {f3(ts['f1_test_optimal_unadapted'])} / {f3(ts['f1_test_optimal_adapted'])} | "
                 f"{f3(r['A_t_src']['diff_vs_U_kept_rate_100']['f1'])} {cis(r['A_t_src']['diff_vs_U_kept_rate_100']['ci95'])} | "
                 f"{f3(r['A_kept_rate_100']['diff_vs_U_kept_rate_100']['f1'])} {cis(r['A_kept_rate_100']['diff_vs_U_kept_rate_100']['ci95'])} | "
                 f"{f3(r['A_t_src']['regret_vs_U_oracle'])} / {f3(r['A_kept_rate_100']['regret_vs_U_oracle'])} / {f3(r['U_kept_rate_100']['regret_vs_U_oracle'])} | "
                 f"{f3(r['A_t_src'].get('LaECE_0'))} / {f3(r['A_kept_rate_100'].get('LaECE_0'))} / {f3(r['A_oracle'].get('LaECE_0'))} | {C[cn]['seconds']} |")
    L += ["", "## Image-level alarm metrics (mean over cells; image-bootstrap CI of the mean)", "",
          "| group | method | sensitivity | CI | FAR | CI |", "|---|---|---|---|---|---|"]
    for g, s in summary.items():
        for m in METHODS:
            v = s[m]
            L.append(f"| {g} | {m} | {f3(v['sensitivity_mean'])} | {cis(v['sensitivity_mean_ci95_image_bootstrap'])} | {f3(v['false_alarm_rate_mean'])} | {cis(v['false_alarm_rate_mean_ci95_image_bootstrap'])} |")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:12]))
    print("artefact ->", OUT_JSON, "and .md", f"({out['summary']['seconds_total']} s)")


if __name__ == "__main__":
    main()
