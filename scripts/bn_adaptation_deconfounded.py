"""Deconfounded test-time batch-norm adaptation + Tent, on the 34 convolutional label-free cells.

Why: bn_adaptation_baseline.py (Schneider et al. 2020 BN re-estimation on 100 unlabelled TARGET frames)
recorded a control on one cell showing that re-estimating the BN statistics on 100 clean SOURCE images
already moves the model (v8m_dfire_s3407 -> pyro_sdis: F1 at t_src 0.213 -> 0.304). A referee therefore
holds the target-BN baseline to be confounded (BN re-estimation per se vs target information) and asks for
Tent (Wang et al., ICLR 2021). This script separates the two effects on every cell and adds Tent.

Reuse: every data / matching / F1 / bootstrap / kept-rate / export function is IMPORTED unchanged from
bn_adaptation_baseline.py (bb), label_free_threshold_refine.py (lr) and operating_points.py (op). The
34 cells, the 100 target frames (refine draw sequence replayed, reproduction asserted), the bootstrap
weights (B = 1000, seed 0, multinomial over target-test images) and the export settings (conf 0.001,
max_det 300, imgsz 640, NMS IoU 0.7, batch-1 predictor, Conv+BN fused by the predictor at export) are
identical to the baseline artefact. No existing file is modified; new detections go to
analysis/detections_bndeconf/<run>__<tag>__<state>.bbox.json (cached: an existing file is re-used).

Model states per cell (all from runs/<run>/weights/best.pt loaded UNFUSED):
  U        unadapted (existing detections/ files; source calval, target calval, target test)
  S        SOURCE-BN: BatchNorm2d running statistics reset, momentum = None (cumulative average), BN in
           train mode alone, forward under no_grad over 100 clean images drawn with numpy default_rng(0)
           from the run's OWN source calval split (d_fire_calval / d_fire_calval_dedup / pyro_sdis_calval,
           as the cell's source_calval; the SAME draw as the baseline's source control), letterboxed
           exactly as the predictor (per-image LetterBox(640, auto=True, stride=32)), batches <= 16 grouped by
           shape. The state depends on the run only, so it is computed once per run and shared by its two
           cells. Exported: target test, target calval (contains the 100 frames), source calval.
  T_reset  TARGET-BN from reset on the 100 target frames = the baseline's adapted model (bb.adapt_bn).
           Target-test / target-calval detections are the baseline's detections_bnadapt/ files (the
           baseline's A_* numbers are re-derived from them and asserted equal); the source calval is
           exported under the reconstructed state (and on the first cell the target test is re-exported
           and asserted identical to the baseline file).
  T_from_S TARGET-BN started from the S state WITHOUT reset: momentum = None continues the cumulative
           average, so the statistics are the batch-count-weighted mean over the 100 source + 100 target
           batches (a source/target mixture). Both variants are run because they are cheap; T_reset is
           the pure target-statistics variant and the one the baseline reported.
  Tent_U   Tent from the unadapted model: BN running stats reset + momentum = None + BN train mode (batch
           statistics are used in the forward passes, the running statistics accumulate as a side effect,
           as in Tent's reference implementation with the modulation parameters updated); ONE epoch over
           the 100 target frames = one pass over the shape-grouped batches (<= 16 images; batch order
           shuffled with numpy default_rng(0)); Adam(lr = 1e-3, betas (0.9, 0.999), no weight decay) on the
           BatchNorm2d affine parameters (gamma, beta) ONLY, every other parameter frozen, fp32.
           Loss (recorded choice): the model is a multi-label sigmoid detector (YOLOv8/11; nc = 1 for the
           Pyro-SDIS-trained runs, so a softmax entropy would be identically 0), so the entropy of each
           candidate is the sum over classes of the BERNOULLI entropy of the sigmoid class probability,
           taken from the raw eval-mode head output (B, 4 + nc, N_anchors) BEFORE NMS; the candidates are
           the top-k (k = 100) anchors per image by max class probability (selection detached, chosen so
           that the loss acts on the anchors NMS/thresholding actually see instead of the ~5000
           near-zero background anchors); mean over candidates and images. After the epoch every BN goes
           back to eval and the predictor fuses the adapted (stats, gamma, beta) into the conv weights.
  Tent_S   Tent started from the S state without reset (stats continue as in T_from_S) — "Tent from
           source-BN", run because it is cheap.
Thresholds evaluated on the target test for every adapted state st (box F1 at IoU 0.5; image-level
sensitivity / false-alarm rate; regret = F1(oracle) - F1(method)):
  {st}_t_src                        the shipped source threshold (unchanged)
  {st}_t_src_prime                  t_src' = F1-optimal on the SOURCE calval under the state's model
                                    (source-calval detections exported under the state)
  {st}_kept_rate_100                PRIMARY kept-rate rule fully re-calibrated under the state: kappa' =
                                    kept detections / image on the state's source calval at t_src',
                                    target threshold from the state's scores on the 100 target frames
  {st}_kept_rate_100_kappa_at_t_src kappa from the state's source calval at the ORIGINAL t_src
  {st}_kept_rate_100_kappa_unadapted the baseline's convention (kappa from the UNADAPTED source at t_src;
                                    for T_reset this reproduces the baseline's A_kept_rate_100 exactly)
  {st}_kept_rate_full               kappa' matched on the full target calval (extra)
  {st}_oracle                       labelled F1-optimal threshold refitted on the state's target calval
plus U_t_src, U_kept_rate_100, U_kept_rate_full, U_oracle re-derived from the existing files (asserted
equal to the refine / baseline artefacts). Regret is reported against the UNADAPTED oracle (primary,
comparable across states) and against each state's own oracle. Bootstrap: image-level, B = 1000, seed 0
(identical weights to the baseline); group CIs = percentile CI of the mean over cells of the replicate
regret (image bootstrap) and a cell-level bootstrap. Wins / losses of every variant vs U_kept_rate_100
and vs U_t_src per group, with CI-significant counts.
Sanity (asserted): (i) unadapted re-export of the first cell's target test equals the existing
detections file EXACTLY (list equality); (ii) S on v8m_dfire_s3407__to__pyro_sdis_caltest reproduces the
baseline's source-control F1 at t_src (0.3038) within 1e-3; (iii) T_reset re-derives the baseline's
A_t_src / A_kept_rate_100 / A_oracle within 1e-9 and (first cell) its re-exported target test equals the
baseline's detections_bnadapt file exactly; (iv) Tent changes gamma/beta (max |delta| recorded) and the
entropy falls from the first to the last step (recorded, not asserted).
Timing: per cell and per state. If a cell exceeds CELL_TIME_LIMIT_S (360 s) the Tent epoch is halved for
the remaining cells and the fact is recorded in "problems". LaECE is not recomputed here.
Writes analysis/bn_adaptation_deconfounded.json (+ .md). GPU (device 0), fp32.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bn_adaptation_baseline as bb  # noqa: E402  (read-only reuse)
import operating_points as op  # noqa: E402
import label_free_threshold_refine as lr  # noqa: E402

BASELINE = HERE / "bn_adaptation_baseline.json"
REFINE = HERE / "label_free_threshold_refine.json"
OUT_DETS = HERE / "detections_bndeconf"
OUT_JSON = HERE / "bn_adaptation_deconfounded.json"
OUT_MD = HERE / "bn_adaptation_deconfounded.md"
SEED, N_ADAPT, B = bb.SEED, bb.N_ADAPT, bb.B
CELL_TIME_LIMIT_S = 360
TENT = {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.0, "batch": bb.BATCH_ADAPT, "epochs": 1.0, "topk": 100, "eps": 1e-7,
        "params": "BatchNorm2d weight (gamma) and bias (beta) only", "entropy": "sum over classes of Bernoulli entropy of sigmoid class prob; top-k anchors per image by max class prob; mean",
        "forward": "eval-mode raw head output (B, 4+nc, N) before NMS; BN modules in train mode (batch statistics), running stats cumulative (momentum=None)",
        "batch_order": "numpy default_rng(0) permutation of the shape-grouped batches", "precision": "fp32"}
STATES = ["S", "T_reset", "T_from_S", "Tent_U", "Tent_S"]
STATE_DESC = {"S": "source-BN (stats re-estimated on 100 source-calval images)", "T_reset": "target-BN from reset on the 100 target frames (= baseline adapted model)",
              "T_from_S": "target-BN continued from the S state (source+target mixture statistics)", "Tent_U": "Tent from the unadapted model (reset stats + affine entropy minimisation)",
              "Tent_S": "Tent from the S state (no reset)"}
VARIANTS = ["t_src", "t_src_prime", "kept_rate_100", "kept_rate_100_kappa_at_t_src", "kept_rate_100_kappa_unadapted", "kept_rate_full", "oracle"]
U_METHODS = ["U_t_src", "U_kept_rate_100", "U_kept_rate_full", "U_oracle"]
HEADLINE = ["U_t_src", "U_kept_rate_100", "U_oracle"] + [f"{st}_{v}" for st in STATES for v in ("t_src", "t_src_prime", "kept_rate_100", "oracle")]
GROUPS = bb.GROUPS


# ----------------------------------------------------------------------------- model states
def fresh(weights, sd=None):
    y, net, n_bn = bb.load_unfused(weights)
    if sd is not None:
        net.load_state_dict(sd)
    return y, net, n_bn


def snapshot(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def bn_continue(net, paths, device):
    """Like bb.adapt_bn but WITHOUT resetting: momentum=None continues the cumulative average from the
    loaded num_batches_tracked, i.e. the statistics become the batch-weighted mean over old + new batches."""
    import torch
    net.to(device).eval()
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    n0 = {int(m.num_batches_tracked.item()) for m in bns}
    assert len(n0) == 1 and next(iter(n0)) > 0, n0
    for m in bns:
        m.momentum = None
        m.train()
    batches = bb.letterbox_batches(paths, bb.BATCH_ADAPT)
    with torch.no_grad():
        for t in batches:
            net(t.to(device))
    for m in bns:
        m.eval()
    n1 = {int(m.num_batches_tracked.item()) for m in bns}
    assert n1 == {next(iter(n0)) + len(batches)}, (n0, n1)
    assert not any(m.training for m in net.modules())
    return {"n_bn": len(bns), "n_batches_prior": next(iter(n0)), "n_batches": len(batches), "batch_sizes": [int(t.shape[0]) for t in batches],
            "batch_shapes": sorted({tuple(t.shape[2:]) for t in batches}), "n_images": len(paths), "reset": False}


def tent(net, paths, device, reset, epochs, seed=SEED):
    """Tent (Wang et al. 2021) for a sigmoid detector — see the module docstring for the recipe."""
    import torch
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    n0 = {int(m.num_batches_tracked.item()) for m in bns}
    params = []
    for m in bns:
        if reset:
            m.reset_running_stats()
        m.momentum = None
        m.train()
        m.weight.requires_grad_(True)
        m.bias.requires_grad_(True)
        params += [m.weight, m.bias]
    g0 = torch.cat([m.weight.detach().flatten() for m in bns]).clone()
    b0 = torch.cat([m.bias.detach().flatten() for m in bns]).clone()
    opt = torch.optim.Adam(params, lr=TENT["lr"], betas=TENT["betas"], weight_decay=TENT["weight_decay"])
    batches = bb.letterbox_batches(paths, TENT["batch"])
    order = np.random.default_rng(seed).permutation(len(batches))
    n_steps = int(math.ceil(len(batches) * epochs))
    losses, n_anchor = [], []
    for bi in order[:n_steps]:
        t = batches[bi].to(device)
        y = net(t)
        y = y[0] if isinstance(y, (tuple, list)) else y
        p = y[:, 4:, :]                                            # sigmoid class probabilities, (B, nc, N)
        k = min(TENT["topk"], p.shape[2])
        idx = p.max(1).values.topk(k, dim=1).indices               # (B, k), detached selection
        ps = p.gather(2, idx.unsqueeze(1).expand(-1, p.shape[1], -1)).clamp(TENT["eps"], 1 - TENT["eps"])
        H = -(ps * ps.log() + (1 - ps) * (1 - ps).log()).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        H.backward()
        opt.step()
        losses.append(float(H.item()))
        n_anchor.append(int(p.shape[2]))
    for m in bns:
        m.eval()
        m.weight.requires_grad_(False)
        m.bias.requires_grad_(False)
    assert not any(m.training for m in net.modules())
    g1 = torch.cat([m.weight.detach().flatten() for m in bns])
    b1 = torch.cat([m.bias.detach().flatten() for m in bns])
    n1 = {int(m.num_batches_tracked.item()) for m in bns}
    assert n1 == {(0 if reset else next(iter(n0))) + n_steps}, (n0, n1, n_steps)
    return {"n_bn": len(bns), "n_affine_params": int(sum(p.numel() for p in params)), "reset": reset, "epochs": epochs, "n_batches": len(batches),
            "n_steps": n_steps, "batch_sizes": [int(batches[i].shape[0]) for i in order[:n_steps]], "batch_shapes": sorted({tuple(t.shape[2:]) for t in batches}),
            "n_images": len(paths), "n_anchors_per_image": sorted(set(n_anchor)), "topk": TENT["topk"], "loss_steps": losses,
            "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
            "gamma_max_abs_delta": float((g1 - g0).abs().max()), "beta_max_abs_delta": float((b1 - b0).abs().max()),
            "gamma_mean_abs_delta": float((g1 - g0).abs().mean()), "beta_mean_abs_delta": float((b1 - b0).abs().mean()),
            "n_batches_tracked_after": next(iter(n1))}


def export_cached(weights, sd, tag, path, state):
    """bb.export (identical loop to export_predictions.py) from a FRESH unfused model carrying `sd`; cached."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    y, net, _ = fresh(weights, sd)
    d = bb.export(y, tag, path)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    meta.update({"bn_adapted": True, "state": state, "state_desc": STATE_DESC[state], "script": Path(__file__).name})
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    del y, net
    return d, False


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter on cell name")
    ap.add_argument("--B", type=int, default=B)
    ap.add_argument("--tent-epochs", type=float, default=TENT["epochs"])
    ap.add_argument("--skip-states", default="", help="comma list among T_from_S,Tent_S to skip (S, T_reset, Tent_U always run)")
    args = ap.parse_args()
    import torch
    skip = {s for s in args.skip_states.split(",") if s}
    assert skip <= {"T_from_S", "Tent_S"}, skip
    states = [s for s in STATES if s not in skip]
    t_start = time.time()
    rj = json.loads(REFINE.read_text(encoding="utf-8"))
    bj = json.loads(BASELINE.read_text(encoding="utf-8"))
    cells = [(n, c) for n, c in rj["cells"].items() if c["family"] == "yolo"]
    if args.only:
        cells = [(n, c) for n, c in cells if args.only in n]
    device = torch.device("cuda:0")
    print(f"{len(cells)} YOLO cells; states {states}; device {torch.cuda.get_device_name(0)}", flush=True)
    OUT_DETS.mkdir(exist_ok=True)

    gt_cache, W_cache, S_cache = {}, {}, {}

    def gt(tag):
        if tag not in gt_cache:
            gt_cache[tag] = op.load_gt(tag)
        return gt_cache[tag]

    def weights_boot(tag, n_img):
        if tag not in W_cache:
            rng = np.random.default_rng(SEED)
            W_cache[tag] = rng.multinomial(n_img, np.full(n_img, 1.0 / n_img), size=args.B).astype(float)
        assert W_cache[tag].shape == (args.B, n_img)
        return W_cache[tag]

    tent_epochs = args.tent_epochs
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "protocol": __doc__, "refine_artefact": str(REFINE), "baseline_artefact": str(BASELINE),
           "baseline_created": bj["created"], "states": states, "state_desc": STATE_DESC, "variants": VARIANTS,
           "config": {"n_adapt_images": N_ADAPT, "seed": SEED, "source_draw": "numpy default_rng(0).choice(n_source_calval_images, 100, replace=False) on the source calval (gt order) — identical to the baseline source control",
                      "target_draw": bb.__doc__.split("unlabelled adaptation set :")[1].split("BN adaptation")[0].strip(),
                      "bn": {"momentum": None, "batch_max": bb.BATCH_ADAPT, "preprocess": "per-image LetterBox(640, auto=True, stride=32), BGR->RGB, /255; batches grouped by letterboxed shape"},
                      "tent": {**TENT, "epochs_requested": args.tent_epochs}, "export": bb.EXPORT, "nms_iou": 0.7,
                      "bootstrap": {"B": args.B, "seed": SEED, "unit": "target-test image (multinomial weights, identical to the refine / baseline artefacts)"},
                      "ultralytics": __import__("ultralytics").__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "cell_time_limit_s": CELL_TIME_LIMIT_S},
           "sanity": {}, "cells": {}, "problems": []}
    reps, rows = {}, []

    for k, (name, c) in enumerate(cells):
        t0 = time.time()
        stage = {}
        run, src, cal, tgt = c["run"], c["source_calval"], c["target_calval_unlabelled"], c["target_test"]
        cls_t, cls_s = set(c["classes_evaluated"]), set(c["classes_source"])
        t_src = c["source"]["t_src"]
        kappa_U = c["source"]["statistics_at_t_src"]["M1c_kept_rate"]
        bc = bj["cells"].get(name)
        s_ids, s_gt, _ = gt(src); u_ids, u_gt, _ = gt(cal); x_ids, x_gt, _ = gt(tgt)
        weights_path = HERE / "runs" / run / "weights" / "best.pt"
        # --- unadapted side (existing files; must reproduce refine + baseline) -------------------------------
        S_U = lr.match_indexed(s_ids, s_gt, op.load_dets(c["files"]["source_calval"]), cls_s)
        U_U = lr.match_indexed(u_ids, u_gt, op.load_dets(c["files"]["target_calval"]), cls_t)
        X_U = lr.match_indexed(x_ids, x_gt, op.load_dets(c["files"]["target_test"]), cls_t)
        t_chk, f1_src_U = bb.oracle_threshold(S_U)
        assert abs(t_chk - t_src) < 1e-9, (name, t_chk, t_src)
        assert abs(lr.source_statistic(lr.unlabelled_view(S_U), t_src, "M1c_kept_rate") - kappa_U) < 1e-12
        pick = bb.replay_draw(U_U["n_img"])
        adapt_ids = [int(u_ids[i]) for i in pick]
        cal_meta = dict(bb.gt_images(cal))
        adapt_paths = [cal_meta[i] for i in adapt_ids]
        if bc is not None:
            assert adapt_ids == bc["adaptation_image_ids"], name
        t_kr_u100 = bb.kept_rate_threshold(bb.subset_view(U_U, pick), kappa_U)
        ref_sub = c["unlabelled_data_sensitivity"]["100"]["M1c_kept_rate"]
        assert t_kr_u100 == ref_sub["threshold_draws"][0], (name, t_kr_u100)
        t_kr_ufull = bb.kept_rate_threshold(lr.unlabelled_view(U_U), kappa_U)
        t_orc_u, f1_orc_u_cal = bb.oracle_threshold(U_U)
        W = weights_boot(tgt, X_U["n_img"])
        src_pick = np.random.default_rng(SEED).choice(len(s_ids), N_ADAPT, replace=False)
        src_meta = dict(bb.gt_images(src))
        src_paths = [src_meta[int(s_ids[i])] for i in src_pick]
        sanity = {}
        # --- sanity (i): unadapted re-export identical to the existing file (first cell) ----------------------
        if k == 0:
            t_s = time.time()
            y0, _, _ = fresh(weights_path)
            d_re = bb.export(y0, tgt, None)
            d_ex = json.loads(Path(c["files"]["target_test"]).read_text(encoding="utf-8"))
            same = d_re == d_ex
            sanity["unadapted_reexport"] = {"cell": name, "n_existing": len(d_ex), "n_reexported": len(d_re), "identical": same, "seconds": round(time.time() - t_s, 1)}
            print(f"[sanity i] {name}: unadapted re-export identical to existing file = {same} ({len(d_ex)} vs {len(d_re)} dets)", flush=True)
            assert same, "unadapted re-export differs from the existing detections file"
            del y0
        # --- S: source-BN (per run) ---------------------------------------------------------------------------
        t_st = time.time()
        if run not in S_cache:
            y, net, n_bn = fresh(weights_path)
            info = bb.adapt_bn(net, src_paths, device)
            S_cache[run] = {"sd": snapshot(net), "info": info, "src_ids": [int(s_ids[i]) for i in src_pick], "src_calval": src, "n_bn": n_bn}
            del y, net
        sdS, n_bn = S_cache[run]["sd"], S_cache[run]["n_bn"]
        if bc is not None:
            sc = bj["sanity"].get("source_control") or {}
            if sc.get("source_calval") == src and bj["sanity"].get("cell") == name:
                assert S_cache[run]["src_ids"] == sc["image_ids"], "source draw differs from the baseline control"
        state_dets, state_info, state_files = {}, {}, {}
        d_src_S, cached = export_cached(weights_path, sdS, src, OUT_DETS / f"{run}__{src}__S.bbox.json", "S")
        d_cal_S, _ = export_cached(weights_path, sdS, cal, OUT_DETS / f"{run}__{cal}__S.bbox.json", "S")
        d_tst_S, _ = export_cached(weights_path, sdS, tgt, OUT_DETS / f"{run}__{tgt}__S.bbox.json", "S")
        state_dets["S"] = (d_src_S, d_cal_S, d_tst_S)
        state_info["S"] = {**S_cache[run]["info"], "reset": True, "shared_across_cells_of_run": True, "source_image_ids": S_cache[run]["src_ids"]}
        stage["S"] = round(time.time() - t_st, 1)
        # --- T_reset: baseline adapted model (target files from detections_bnadapt; source calval exported) -----
        t_st = time.time()
        y, net, _ = fresh(weights_path)
        info = bb.adapt_bn(net, adapt_paths, device)
        sdT = snapshot(net); del y, net
        f_tst_T, f_cal_T = bb.OUT_DETS / f"{run}__{tgt}.bbox.json", bb.OUT_DETS / f"{run}__{cal}.bbox.json"
        assert f_tst_T.exists() and f_cal_T.exists(), (f_tst_T, f_cal_T)
        d_tst_T = json.loads(f_tst_T.read_text(encoding="utf-8")); d_cal_T = json.loads(f_cal_T.read_text(encoding="utf-8"))
        if k == 0:
            t_s = time.time()
            y, net, _ = fresh(weights_path, sdT)
            d_re = bb.export(y, tgt, None); del y, net
            same = d_re == d_tst_T
            sanity["T_reset_reexport"] = {"cell": name, "n_baseline_file": len(d_tst_T), "n_reexported": len(d_re), "identical": same, "seconds": round(time.time() - t_s, 1)}
            print(f"[sanity iii] {name}: T_reset re-export identical to detections_bnadapt file = {same}", flush=True)
            assert same, "T_reset reconstruction does not reproduce the baseline adapted detections"
        d_src_T, _ = export_cached(weights_path, sdT, src, OUT_DETS / f"{run}__{src}__T_reset.bbox.json", "T_reset")
        state_dets["T_reset"] = (d_src_T, d_cal_T, d_tst_T)
        state_info["T_reset"] = {**info, "reset": True, "target_files_from": str(bb.OUT_DETS)}
        stage["T_reset"] = round(time.time() - t_st, 1)
        # --- T_from_S ------------------------------------------------------------------------------------------
        if "T_from_S" in states:
            t_st = time.time()
            y, net, _ = fresh(weights_path, sdS)
            info = bn_continue(net, adapt_paths, device)
            sd = snapshot(net); del y, net
            ds = [export_cached(weights_path, sd, tag, OUT_DETS / f"{run}__{tag}__T_from_S.bbox.json", "T_from_S")[0] for tag in (src, cal, tgt)]
            state_dets["T_from_S"] = tuple(ds); state_info["T_from_S"] = info
            stage["T_from_S"] = round(time.time() - t_st, 1)
        # --- Tent_U / Tent_S ------------------------------------------------------------------------------------
        for st, sd0, reset in (("Tent_U", None, True), ("Tent_S", sdS, False)):
            if st not in states:
                continue
            t_st = time.time()
            y, net, _ = fresh(weights_path, sd0)
            t_t = time.time()
            info = tent(net, adapt_paths, device, reset=reset, epochs=tent_epochs)
            info["seconds"] = round(time.time() - t_t, 1)
            sd = snapshot(net); del y, net
            torch.cuda.empty_cache()
            ds = [export_cached(weights_path, sd, tag, OUT_DETS / f"{run}__{tag}__{st}.bbox.json", st)[0] for tag in (src, cal, tgt)]
            state_dets[st] = tuple(ds); state_info[st] = info
            stage[st] = round(time.time() - t_st, 1)
            assert info["gamma_max_abs_delta"] > 0 or info["beta_max_abs_delta"] > 0, f"{name} {st}: Tent did not change the affine parameters"
        for st in states:
            state_files[st] = {tag: str(OUT_DETS / f"{run}__{tag}__{st}.bbox.json") for tag in (src, cal, tgt)}
        state_files["T_reset"].update({cal: str(f_cal_T), tgt: str(f_tst_T)})
        secs_gpu = round(time.time() - t0, 1)
        # --- thresholds and evaluation ---------------------------------------------------------------------------
        t_ev = time.time()
        thr = {"U_t_src": (X_U, t_src), "U_kept_rate_100": (X_U, t_kr_u100), "U_kept_rate_full": (X_U, t_kr_ufull), "U_oracle": (X_U, t_orc_u)}
        own = {m: "U_oracle" for m in U_METHODS}
        src_side = {"U": {"t_src": t_src, "t_src_prime": t_src, "f1_src_at_t_src": f1_src_U, "f1_src_at_t_src_prime": f1_src_U, "kappa_at_t_src": kappa_U, "kappa_at_t_src_prime": kappa_U}}
        thresholds = {"U": {"t_src": t_src, "kept_rate_100": t_kr_u100, "kept_rate_full": t_kr_ufull, "oracle": t_orc_u}}
        cal_orc = {"U": f1_orc_u_cal}
        test_opt = {"U": float(op.pr_curve(X_U["score"], X_U["tp"], X_U["n_gt"])[3].max())}
        n_dets = {"U": {"source": int(len(S_U["score"])), "target_calval": int(len(U_U["score"])), "target_test": int(len(X_U["score"]))}}
        for st in states:
            d_src, d_cal, d_tst = state_dets[st]
            Ss = lr.match_indexed(s_ids, s_gt, bb.dets_by_img(d_src), cls_s)
            Us = lr.match_indexed(u_ids, u_gt, bb.dets_by_img(d_cal), cls_t)
            Xs = lr.match_indexed(x_ids, x_gt, bb.dets_by_img(d_tst), cls_t)
            Sc = op.pr_curve(Ss["score"], Ss["tp"], Ss["n_gt"])
            t_prime, f1_prime = bb.oracle_threshold(Ss)
            f1_src_t = op.at_threshold(*Sc, t_src)["f1"]
            sv = lr.unlabelled_view(Ss)
            kappa_t = lr.source_statistic(sv, t_src, "M1c_kept_rate")
            kappa_p = lr.source_statistic(sv, t_prime, "M1c_kept_rate")
            sub = bb.subset_view(Us, pick)
            t_kr = bb.kept_rate_threshold(sub, kappa_p)
            t_kr_t = bb.kept_rate_threshold(sub, kappa_t)
            t_kr_kU = bb.kept_rate_threshold(sub, kappa_U)
            t_kr_full = bb.kept_rate_threshold(lr.unlabelled_view(Us), kappa_p)
            t_orc, f1_orc_cal = bb.oracle_threshold(Us)
            thr.update({f"{st}_t_src": (Xs, t_src), f"{st}_t_src_prime": (Xs, t_prime), f"{st}_kept_rate_100": (Xs, t_kr),
                        f"{st}_kept_rate_100_kappa_at_t_src": (Xs, t_kr_t), f"{st}_kept_rate_100_kappa_unadapted": (Xs, t_kr_kU),
                        f"{st}_kept_rate_full": (Xs, t_kr_full), f"{st}_oracle": (Xs, t_orc)})
            for v in VARIANTS:
                own[f"{st}_{v}"] = f"{st}_oracle"
            src_side[st] = {"t_src": t_src, "t_src_prime": t_prime, "f1_src_at_t_src": f1_src_t, "f1_src_at_t_src_prime": f1_prime,
                            "kappa_at_t_src": kappa_t, "kappa_at_t_src_prime": kappa_p, "kappa_unadapted": kappa_U}
            thresholds[st] = {"t_src": t_src, "t_src_prime": t_prime, "kept_rate_100": t_kr, "kept_rate_100_kappa_at_t_src": t_kr_t,
                              "kept_rate_100_kappa_unadapted": t_kr_kU, "kept_rate_full": t_kr_full, "oracle": t_orc}
            cal_orc[st] = f1_orc_cal
            test_opt[st] = float(op.pr_curve(Xs["score"], Xs["tp"], Xs["n_gt"])[3].max())
            n_dets[st] = {"source": int(len(Ss["score"])), "target_calval": int(len(Us["score"])), "target_test": int(len(Xs["score"]))}
        res, rep = {}, {}
        for m, (XX, t) in thr.items():
            point, r = lr.evaluate(XX, t, W)
            res[m] = {"threshold": t, "state": next(s for s in ["U"] + STATES if m.startswith(s + "_")), **point}
            rep[m] = r
        # reproduction of the refine and baseline artefacts
        rm = c["methods"]
        for m, rk in (("U_t_src", "M0_t_src"), ("U_kept_rate_full", "M1c_kept_rate"), ("U_oracle", "ORC_target_calval")):
            assert abs(res[m]["f1"] - rm[rk]["f1"]) < 1e-9, (name, m)
        assert abs((res["U_oracle"]["f1"] - res["U_kept_rate_100"]["f1"]) - ref_sub["regret_draws"][0]) < 1e-9, name
        repro_T = None
        if bc is not None:
            bm = bc["methods"]
            pairs = (("T_reset_t_src", "A_t_src"), ("T_reset_kept_rate_100_kappa_unadapted", "A_kept_rate_100"), ("T_reset_oracle", "A_oracle"), ("U_kept_rate_100", "U_kept_rate_100"))
            repro_T = {m: {"this": res[m]["f1"], "baseline": bm[bk]["f1"], "abs_diff": abs(res[m]["f1"] - bm[bk]["f1"])} for m, bk in pairs}
            for m, bk in pairs:
                assert abs(res[m]["f1"] - bm[bk]["f1"]) < 1e-9, (name, m, res[m]["f1"], bm[bk]["f1"])
            assert abs(thresholds["T_reset"]["kept_rate_100_kappa_unadapted"] - bc["thresholds"]["kept_rate_100_adapted"]) < 1e-12
        if name == "v8m_dfire_s3407__to__pyro_sdis_caltest":
            ref = bj["sanity"]["source_control"]["f1_at_t_src"]
            sanity["source_bn_control"] = {"cell": name, "f1_at_t_src_S": res["S_t_src"]["f1"], "baseline_source_control": ref, "abs_diff": abs(res["S_t_src"]["f1"] - ref),
                                           "passed_1e-3": abs(res["S_t_src"]["f1"] - ref) <= 1e-3, "f1_at_t_src_U": res["U_t_src"]["f1"]}
            print(f"[sanity ii] {name}: S F1@t_src {res['S_t_src']['f1']:.4f} vs baseline source-control {ref:.4f}", flush=True)
            assert abs(res["S_t_src"]["f1"] - ref) <= 1e-3, "source-BN does not reproduce the baseline control"
        f1_orc_u = res["U_oracle"]["f1"]
        for m in thr:
            r = res[m]
            o = own[m]
            r["own_oracle"] = o
            r["regret_vs_U_oracle"] = f1_orc_u - r["f1"]
            r["regret_vs_own_oracle"] = res[o]["f1"] - r["f1"]
            r["ci95"] = {q: lr.ci(rep[m][q]) for q in ("f1", "sensitivity", "false_alarm_rate")}
            r["regret_vs_U_oracle_ci95"] = lr.ci(rep["U_oracle"]["f1"] - rep[m]["f1"])
            r["regret_vs_own_oracle_ci95"] = lr.ci(rep[o]["f1"] - rep[m]["f1"])
            rep[m]["regret_U"] = rep["U_oracle"]["f1"] - rep[m]["f1"]
            rep[m]["regret_own"] = rep[o]["f1"] - rep[m]["f1"]
            for ref_m in ("U_t_src", "U_kept_rate_100"):
                if m == ref_m:
                    continue
                d = rep[m]["f1"] - rep[ref_m]["f1"]
                rep[m][f"diff_vs_{ref_m}"] = d
                r[f"diff_vs_{ref_m}"] = {"f1": r["f1"] - res[ref_m]["f1"], "ci95": lr.ci(d), "p_boot_le_0": float(np.mean(d <= 0)),
                                         "sensitivity": (r["sensitivity"] - res[ref_m]["sensitivity"]) if (r["sensitivity"] is not None and res[ref_m]["sensitivity"] is not None) else None,
                                         "false_alarm_rate": (r["false_alarm_rate"] - res[ref_m]["false_alarm_rate"]) if (r["false_alarm_rate"] is not None and res[ref_m]["false_alarm_rate"] is not None) else None}
        secs_eval = round(time.time() - t_ev, 1)
        secs = round(time.time() - t0, 1)
        out["sanity"].update(sanity)
        cell = {"run": run, "model": c["model"], "train_set": c["train_set"], "seed": c["seed"], "target": c["target"], "target_test": tgt,
                "source_calval": src, "target_calval_unlabelled": cal, "classes_evaluated": sorted(cls_t), "classes_source": sorted(cls_s),
                "weights": str(weights_path), "n_bn_modules": n_bn, "adaptation_image_ids": adapt_ids, "source_bn_image_ids": S_cache[run]["src_ids"],
                "states": {st: {"desc": STATE_DESC[st], "adaptation": state_info[st], "files": state_files[st], "source": src_side[st], "thresholds": thresholds[st],
                                "calval_oracle_f1": cal_orc[st], "f1_test_optimal": test_opt[st], "n_dets": n_dets[st], "seconds": stage[st]} for st in states},
                "unadapted": {"files": c["files"], "source": src_side["U"], "thresholds": thresholds["U"], "calval_oracle_f1": cal_orc["U"], "f1_test_optimal": test_opt["U"], "n_dets": n_dets["U"]},
                "baseline_reproduction": repro_T, "tent_epochs_used": tent_epochs,
                "methods": res, "seconds": secs, "seconds_gpu_incl_sanity": secs_gpu, "seconds_eval": secs_eval, "seconds_by_state": stage}
        out["cells"][name] = cell
        reps[name] = rep
        rows.append(name)
        elapsed = time.time() - t_start
        print(f"[{k + 1}/{len(cells)}] {name}: {secs}s (stages {stage}) | F1 U t_src {res['U_t_src']['f1']:.3f} kr {res['U_kept_rate_100']['f1']:.3f} orc {f1_orc_u:.3f} | "
              f"S t_src {res['S_t_src']['f1']:.3f} t' {res['S_t_src_prime']['f1']:.3f} kr {res['S_kept_rate_100']['f1']:.3f} orc {res['S_oracle']['f1']:.3f} | "
              f"T t_src {res['T_reset_t_src']['f1']:.3f} kr {res['T_reset_kept_rate_100']['f1']:.3f} | "
              + (f"Tent t_src {res['Tent_U_t_src']['f1']:.3f} kr {res['Tent_U_kept_rate_100']['f1']:.3f} orc {res['Tent_U_oracle']['f1']:.3f} (H {state_info['Tent_U']['loss_first']:.4f}->{state_info['Tent_U']['loss_last']:.4f}) | " if "Tent_U" in states else "")
              + f"elapsed {elapsed / 60:.1f} min, projected {elapsed / (k + 1) * len(cells) / 60:.0f} min", flush=True)
        secs_excl_sanity = secs - sum(v.get("seconds", 0.0) for v in sanity.values())
        tent_secs = sum(state_info[st]["seconds"] for st in states if st.startswith("Tent"))
        if secs_excl_sanity > CELL_TIME_LIMIT_S:
            if tent_secs > 0.25 * secs_excl_sanity and tent_epochs > 0.5:
                tent_epochs = 0.5
                out["problems"].append(f"{name} took {secs_excl_sanity}s excl. sanity > {CELL_TIME_LIMIT_S}s with Tent = {tent_secs}s of it; Tent reduced to half an epoch for the remaining cells (from cell index {k + 1})")
                print("NOTE: cell exceeded the time limit and Tent was the slow part; Tent halved for the remaining cells", flush=True)
            else:
                out["problems"].append(f"{name} took {secs_excl_sanity:.1f}s excl. sanity > {CELL_TIME_LIMIT_S}s, but the Tent stage was only {tent_secs:.1f}s "
                                       f"(the cell is export-bound: {sum(n_dets[st]['target_test'] > 0 for st in states)} states x 3 exports); halving Tent would not help, so the full epoch was kept")
                print(f"NOTE: cell exceeded the time limit (export-bound, Tent {tent_secs:.1f}s); full Tent epoch kept", flush=True)
        out["partial"] = True
        OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------------- summaries
    C = out["cells"]
    methods = list(C[rows[0]]["methods"].keys()) if rows else []

    def group_summary(sel, rng_cells):
        s = {}
        for m in methods:
            f1 = np.array([C[c]["methods"][m]["f1"] for c in sel])
            e = {"n_cells": len(sel), "f1_mean": float(f1.mean()), "own_oracle": C[sel[0]]["methods"][m]["own_oracle"]}
            for key, rk in (("U", "regret_vs_U_oracle"), ("own", "regret_vs_own_oracle")):
                reg = np.array([C[c]["methods"][m][rk] for c in sel])
                reg_rep = np.mean([reps[c][m][f"regret_{key}"] for c in sel], axis=0)
                cb = [reg[rng_cells.integers(0, len(sel), len(sel))].mean() for _ in range(args.B)] if len(sel) >= 2 else None
                e[rk] = {"mean": float(reg.mean()), "median": float(np.median(reg)), "max": float(reg.max()), "sd_across_cells": float(reg.std(ddof=1)) if len(sel) > 1 else None,
                         "mean_ci95_image_bootstrap": lr.ci(reg_rep), "mean_ci95_cell_bootstrap": lr.ci(cb) if cb is not None else None}
            for ref_m in ("U_t_src", "U_kept_rate_100"):
                if m == ref_m:
                    continue
                d = np.array([C[c]["methods"][m][f"diff_vs_{ref_m}"]["f1"] for c in sel])
                d_rep = np.mean([reps[c][m][f"diff_vs_{ref_m}"] for c in sel], axis=0)
                e[f"vs_{ref_m}"] = {"f1_gain_mean": float(d.mean()), "f1_gain_ci95_image_bootstrap": lr.ci(d_rep),
                                    "wins": int(np.sum(d > 1e-12)), "losses": int(np.sum(d < -1e-12)), "ties": int(np.sum(np.abs(d) <= 1e-12)),
                                    "wins_ci_excludes_0": int(sum(1 for c in sel if C[c]["methods"][m][f"diff_vs_{ref_m}"]["ci95"][0] > 0)),
                                    "losses_ci_excludes_0": int(sum(1 for c in sel if C[c]["methods"][m][f"diff_vs_{ref_m}"]["ci95"][1] < 0))}
            for q in ("sensitivity", "false_alarm_rate"):
                pts = np.array([v for v in (C[c]["methods"][m][q] for c in sel) if v is not None], dtype=float)
                e[f"{q}_mean"] = float(pts.mean()) if len(pts) else None
                e[f"{q}_mean_ci95_image_bootstrap"] = lr.ci(np.nanmean([reps[c][m][q] for c in sel], axis=0))
            s[m] = e
        return s

    rng_cells = np.random.default_rng(SEED)
    groups = {g: [c for c in rows if f(C[c])] for g, f in GROUPS.items()}
    summary = {g: group_summary(sel, rng_cells) for g, sel in groups.items() if sel}
    # source-side summary: how much does source-BN move the source operating point?
    src_sum = {}
    for st in states:
        d_t = np.array([C[c]["states"][st]["source"]["t_src_prime"] - C[c]["states"][st]["source"]["t_src"] for c in rows])
        d_f1 = np.array([C[c]["states"][st]["source"]["f1_src_at_t_src"] - C[c]["unadapted"]["source"]["f1_src_at_t_src"] for c in rows])
        d_f1p = np.array([C[c]["states"][st]["source"]["f1_src_at_t_src_prime"] - C[c]["unadapted"]["source"]["f1_src_at_t_src"] for c in rows])
        d_k = np.array([C[c]["states"][st]["source"]["kappa_at_t_src"] - C[c]["states"][st]["source"]["kappa_unadapted"] for c in rows])
        src_sum[st] = {"t_src_prime_minus_t_src": {"mean": float(d_t.mean()), "min": float(d_t.min()), "max": float(d_t.max())},
                       "source_f1_at_t_src_minus_unadapted": {"mean": float(d_f1.mean()), "min": float(d_f1.min()), "max": float(d_f1.max())},
                       "source_f1_at_t_src_prime_minus_unadapted_f1_at_t_src": {"mean": float(d_f1p.mean()), "min": float(d_f1p.min()), "max": float(d_f1p.max())},
                       "kappa_at_t_src_minus_kappa_unadapted": {"mean": float(d_k.mean()), "min": float(d_k.min()), "max": float(d_k.max())}}
    # verdict block on the source-BN question (yolo_all)
    verdict = {}
    if "yolo_all" in summary:
        ya = summary["yolo_all"]
        for m in ("S_t_src", "S_t_src_prime", "S_kept_rate_100", "S_oracle"):
            v = ya[m]
            ref_m = "U_kept_rate_100" if "kept_rate" in m else "U_t_src"
            vv = v[f"vs_{ref_m}"]
            verdict[m] = {"reference": ref_m, "f1_gain_mean": vv["f1_gain_mean"], "ci95": vv["f1_gain_ci95_image_bootstrap"], "wins": vv["wins"], "losses": vv["losses"],
                          "wins_ci_sig": vv["wins_ci_excludes_0"], "losses_ci_sig": vv["losses_ci_excludes_0"],
                          "systematic_gain": bool(vv["f1_gain_ci95_image_bootstrap"][0] > 0 and vv["wins"] > 2 * vv["losses"]),
                          "systematic_loss": bool(vv["f1_gain_ci95_image_bootstrap"][1] < 0 and vv["losses"] > 2 * vv["wins"])}
        vo = ya["S_oracle"]["vs_U_t_src"]
        verdict["S_oracle_vs_U_oracle"] = {"f1_gain_mean": ya["S_oracle"]["f1_mean"] - ya["U_oracle"]["f1_mean"], "regret_vs_U_oracle_mean": ya["S_oracle"]["regret_vs_U_oracle"]["mean"],
                                           "regret_vs_U_oracle_ci95_image_bootstrap": ya["S_oracle"]["regret_vs_U_oracle"]["mean_ci95_image_bootstrap"],
                                           "cells_S_oracle_above_U_oracle": int(sum(1 for c in rows if C[c]["methods"]["S_oracle"]["f1"] > C[c]["methods"]["U_oracle"]["f1"] + 1e-12))}
        one = C.get("v8m_dfire_s3407__to__pyro_sdis_caltest")
        if one is not None:
            verdict["control_cell"] = {"cell": "v8m_dfire_s3407__to__pyro_sdis_caltest", "U_t_src": one["methods"]["U_t_src"]["f1"], "S_t_src": one["methods"]["S_t_src"]["f1"],
                                       "S_t_src_prime": one["methods"]["S_t_src_prime"]["f1"], "U_kept_rate_100": one["methods"]["U_kept_rate_100"]["f1"], "S_kept_rate_100": one["methods"]["S_kept_rate_100"]["f1"],
                                       "U_oracle": one["methods"]["U_oracle"]["f1"], "S_oracle": one["methods"]["S_oracle"]["f1"]}
    out["summary"] = {"groups": summary, "group_cells": groups, "source_side": src_sum, "verdict_source_bn": verdict, "n_cells": len(rows), "methods": methods, "headline_methods": [m for m in HEADLINE if m in methods],
                      "seconds_total": round(time.time() - t_start, 1), "cell_seconds_max": max((C[c]["seconds"] for c in rows), default=None),
                      "cell_seconds_mean": float(np.mean([C[c]["seconds"] for c in rows])) if rows else None,
                      "state_seconds_mean": {st: float(np.mean([C[c]["seconds_by_state"][st] for c in rows])) for st in states} if rows else None,
                      "tent_seconds_mean": {st: float(np.mean([C[c]["states"][st]["adaptation"]["seconds"] for c in rows])) for st in states if st.startswith("Tent")} if rows else None,
                      "tent_loss_first_last_mean": {st: [float(np.mean([C[c]["states"][st]["adaptation"]["loss_first"] for c in rows])), float(np.mean([C[c]["states"][st]["adaptation"]["loss_last"] for c in rows]))] for st in states if st.startswith("Tent")} if rows else None}
    out["partial"] = False
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------------- md
    def f3(x):
        return "" if x is None else f"{x:.3f}"

    def cis(c):
        return "" if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]"

    sn = out["sanity"]
    L = ["# Deconfounded BN adaptation (source-BN vs target-BN) and Tent vs the label-free kept-rate rule — machine-written by bn_adaptation_deconfounded.py", "",
         f"Cells: {len(rows)} YOLO (v8 / 11) cross-domain cells from {REFINE.name}; baseline artefact {BASELINE.name}. States: " + "; ".join(f"**{st}** = {STATE_DESC[st]}" for st in states) + ". "
         f"Adaptation sets: 100 target frames = the refine artefact's seed-0 n = 100 draw (asserted); 100 source images = seed-0 draw from the run's own source calval (the baseline's control draw). "
         f"Regret = F1(oracle) - F1(method), box F1 at IoU 0.5 on the target test; primary reference = the UNADAPTED labelled oracle; 'own' = the state's refitted oracle. "
         f"CIs: image-level bootstrap of the target test (B = {args.B}, seed {SEED}); cell-level bootstrap also given. Tent recipe: Adam lr {TENT['lr']}, batch <= {TENT['batch']}, "
         f"{args.tent_epochs} epoch over the 100 frames, BN gamma/beta only, loss = mean over top-{TENT['topk']} anchors/image of the summed per-class Bernoulli entropy of the sigmoid class probabilities (pre-NMS). "
         f"Total {out['summary']['seconds_total']} s; max cell {out['summary']['cell_seconds_max']} s; mean per state (s): {({k: round(v, 1) for k, v in out['summary']['state_seconds_mean'].items()})}.", ""]
    L += ["Sanity: " + "; ".join(f"{k}: " + ", ".join(f"{a} = {(f'{b:.4f}' if isinstance(b, float) else b)}" for a, b in v.items() if a != 'cell') + f" ({v.get('cell')})" for k, v in sn.items()), ""]
    if out["problems"]:
        L += ["**Problems:** " + "; ".join(out["problems"]), ""]
    L += ["Kept-rate variants per state: `{st}_kept_rate_100` (PRIMARY) = kappa' from the state's own source calval at t_src', target threshold from the state's 100 target frames; "
          "`_kappa_at_t_src` = kappa from the state's source calval at the original t_src; `_kappa_unadapted` = the baseline convention (kappa from the unadapted source; for T_reset = baseline A_kept_rate_100).", ""]
    hl = out["summary"]["headline_methods"]
    for g, s in summary.items():
        L += [f"## {g} (n = {s['U_t_src']['n_cells']}) — headline", "",
              "| method | F1 mean | regret vs U-oracle mean | CI (image boot) | CI (cell boot) | median | regret vs own oracle | CI | gain vs U_t_src | CI | W/L (CI-sig) | gain vs U_kept_rate_100 | CI | W/L (CI-sig) | sens | FAR |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in hl:
            v = s[m]
            rU, rO = v["regret_vs_U_oracle"], v["regret_vs_own_oracle"]
            vt, vk = v.get("vs_U_t_src"), v.get("vs_U_kept_rate_100")
            L.append(f"| {m} | {f3(v['f1_mean'])} | {f3(rU['mean'])} | {cis(rU['mean_ci95_image_bootstrap'])} | {cis(rU['mean_ci95_cell_bootstrap'])} | {f3(rU['median'])} | {f3(rO['mean'])} | {cis(rO['mean_ci95_image_bootstrap'])} | "
                     f"{f3(vt['f1_gain_mean']) if vt else ''} | {cis(vt['f1_gain_ci95_image_bootstrap']) if vt else ''} | {f'{vt['wins']}/{vt['losses']} ({vt['wins_ci_excludes_0']}/{vt['losses_ci_excludes_0']})' if vt else ''} | "
                     f"{f3(vk['f1_gain_mean']) if vk else ''} | {cis(vk['f1_gain_ci95_image_bootstrap']) if vk else ''} | {f'{vk['wins']}/{vk['losses']} ({vk['wins_ci_excludes_0']}/{vk['losses_ci_excludes_0']})' if vk else ''} | "
                     f"{f3(v['sensitivity_mean'])} | {f3(v['false_alarm_rate_mean'])} |")
        L += ["", f"### {g} — all kept-rate variants", "", "| method | F1 mean | regret vs U-oracle | CI (image boot) | regret vs own oracle | gain vs U_kept_rate_100 | CI | W/L (CI-sig) | sens | FAR |", "|---|---|---|---|---|---|---|---|---|---|"]
        for m in methods:
            if m in hl or "kept_rate" not in m:
                continue
            v = s[m]; rU, rO, vk = v["regret_vs_U_oracle"], v["regret_vs_own_oracle"], v["vs_U_kept_rate_100"]
            L.append(f"| {m} | {f3(v['f1_mean'])} | {f3(rU['mean'])} | {cis(rU['mean_ci95_image_bootstrap'])} | {f3(rO['mean'])} | {f3(vk['f1_gain_mean'])} | {cis(vk['f1_gain_ci95_image_bootstrap'])} | "
                     f"{vk['wins']}/{vk['losses']} ({vk['wins_ci_excludes_0']}/{vk['losses_ci_excludes_0']}) | {f3(v['sensitivity_mean'])} | {f3(v['false_alarm_rate_mean'])} |")
        L.append("")
    L += ["## Source side: does BN re-estimation move the SOURCE operating point? (per cell)", "",
          "| cell | t_src | U F1 src | " + " | ".join(f"{st} t_src' / F1src@t_src / F1src@t' / kappa@t_src / kappa'" for st in states) + " | kappa U |",
          "|---|---|---|" + "---|" * len(states) + "---|"]
    for cn in rows:
        u = C[cn]["unadapted"]["source"]
        L.append(f"| {cn} | {f3(u['t_src'])} | {f3(u['f1_src_at_t_src'])} | " + " | ".join(
            f"{f3(C[cn]['states'][st]['source']['t_src_prime'])} / {f3(C[cn]['states'][st]['source']['f1_src_at_t_src'])} / {f3(C[cn]['states'][st]['source']['f1_src_at_t_src_prime'])} / "
            f"{C[cn]['states'][st]['source']['kappa_at_t_src']:.2f} / {C[cn]['states'][st]['source']['kappa_at_t_src_prime']:.2f}" for st in states) + f" | {u['kappa_at_t_src']:.2f} |")
    L += ["", "Mean over cells: " + "; ".join(f"{st}: t' - t_src {src_sum[st]['t_src_prime_minus_t_src']['mean']:+.3f}, source F1@t_src {src_sum[st]['source_f1_at_t_src_minus_unadapted']['mean']:+.3f} vs unadapted, "
                                              f"source F1@t' {src_sum[st]['source_f1_at_t_src_prime_minus_unadapted_f1_at_t_src']['mean']:+.3f} vs unadapted F1@t_src" for st in states), ""]
    L += ["## Per cell (F1 on the target test; regret vs U-oracle in brackets)", "",
          "| cell | U t_src | U kr100 | U orc | " + " | ".join(f"{st} t_src | {st} t' | {st} kr100 | {st} orc" for st in states) + " | test-opt U / " + " / ".join(states) + " | s |",
          "|---|---|---|---|" + "---|---|---|---|" * len(states) + "---|---|"]
    for cn in rows:
        r = C[cn]["methods"]

        def cell_f(m):
            return f"{f3(r[m]['f1'])} ({f3(r[m]['regret_vs_U_oracle'])})"
        L.append(f"| {cn} | {cell_f('U_t_src')} | {cell_f('U_kept_rate_100')} | {f3(r['U_oracle']['f1'])} | " + " | ".join(
            f"{cell_f(f'{st}_t_src')} | {cell_f(f'{st}_t_src_prime')} | {cell_f(f'{st}_kept_rate_100')} | {f3(r[f'{st}_oracle']['f1'])}" for st in states)
            + f" | {f3(C[cn]['unadapted']['f1_test_optimal'])} / " + " / ".join(f3(C[cn]["states"][st]["f1_test_optimal"]) for st in states) + f" | {C[cn]['seconds']} |")
    L += ["", "## Tent diagnostics (per cell)", "", "| cell | " + " | ".join(f"{st} steps | {st} H first -> last | {st} max abs d gamma / d beta | {st} s" for st in states if st.startswith("Tent")) + " |",
          "|---|" + "---|---|---|---|" * len([s for s in states if s.startswith("Tent")])]
    for cn in rows:
        L.append(f"| {cn} | " + " | ".join(f"{C[cn]['states'][st]['adaptation']['n_steps']} | {C[cn]['states'][st]['adaptation']['loss_first']:.4f} -> {C[cn]['states'][st]['adaptation']['loss_last']:.4f} | "
                                          f"{C[cn]['states'][st]['adaptation']['gamma_max_abs_delta']:.4f} / {C[cn]['states'][st]['adaptation']['beta_max_abs_delta']:.4f} | {C[cn]['states'][st]['adaptation']['seconds']}"
                                          for st in states if st.startswith("Tent")) + " |")
    L += ["", "## Image-level alarm metrics (mean over cells; image-bootstrap CI of the mean)", "", "| group | method | sensitivity | CI | FAR | CI |", "|---|---|---|---|---|---|"]
    for g, s in summary.items():
        for m in hl:
            v = s[m]
            L.append(f"| {g} | {m} | {f3(v['sensitivity_mean'])} | {cis(v['sensitivity_mean_ci95_image_bootstrap'])} | {f3(v['false_alarm_rate_mean'])} | {cis(v['false_alarm_rate_mean_ci95_image_bootstrap'])} |")
    if verdict:
        L += ["", "## Verdict on source-BN re-estimation (yolo_all; rule: 'systematic' = image-bootstrap CI of the mean gain excludes 0 AND wins > 2 x losses)", ""]
        for m, v in verdict.items():
            if "reference" in v:
                L.append(f"- {m} vs {v['reference']}: mean gain {v['f1_gain_mean']:+.3f} {cis(v['ci95'])}, W/L {v['wins']}/{v['losses']} (CI-sig {v['wins_ci_sig']}/{v['losses_ci_sig']}) -> "
                         f"{'SYSTEMATIC GAIN' if v['systematic_gain'] else ('SYSTEMATIC LOSS' if v['systematic_loss'] else 'NOT systematic')}")
        so = verdict.get("S_oracle_vs_U_oracle")
        if so:
            L.append(f"- S_oracle vs U_oracle: mean F1 difference {so['f1_gain_mean']:+.3f}; S-oracle regret vs U-oracle {so['regret_vs_U_oracle_mean']:+.3f} {cis(so['regret_vs_U_oracle_ci95_image_bootstrap'])}; "
                     f"S-oracle above U-oracle in {so['cells_S_oracle_above_U_oracle']}/{len(rows)} cells")
        cc = verdict.get("control_cell")
        if cc:
            L.append(f"- control cell {cc['cell']}: U t_src {cc['U_t_src']:.3f} -> S t_src {cc['S_t_src']:.3f} (t' {cc['S_t_src_prime']:.3f}); U kept-rate {cc['U_kept_rate_100']:.3f} vs S kept-rate {cc['S_kept_rate_100']:.3f}; "
                     f"oracles U {cc['U_oracle']:.3f} vs S {cc['S_oracle']:.3f}")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:8]))
    print("artefact ->", OUT_JSON, "and .md", f"({out['summary']['seconds_total']} s)")


if __name__ == "__main__":
    main()
