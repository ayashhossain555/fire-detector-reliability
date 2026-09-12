"""Referee-revision analyses for the label-free operating-point transfer (2026-09-13, CPU only, no GPU).

Four independent parts; everything is recomputed from the exported detections / stored scores and
asserted against the existing artefacts where they overlap. No existing file is modified.

A. CAMERA-DISJOINT re-run of the kept-rate rule (M1c) for every cell whose target is Pyro-SDIS.
   Pyro-SDIS val (calval + caltest pooled, 4,099 images, 24 cameras) is split into the 4 camera folds
   of camera_conformal.py (its pool() / camera_folds() are imported unchanged and the fold list is
   asserted equal to camera_conformal.json). Per fold: the unlabelled adaptation set is the
   calibration cameras' detections (scores only) -> t_M1c by label_free_threshold_refine.solve_threshold
   (imported unchanged; kappa = kept detections per image at t_src on the source calval, recomputed and
   asserted equal to the refine artefact); the labelled oracle is the F1-optimal threshold on the
   calibration cameras WITH labels; evaluation is on the held-out cameras: box F1 (IoU 0.5, same greedy
   matching as operating_points.py), image-level sensitivity / false-alarm rate, regret = F1(ORC) - F1(m).
   Also the referee's literal ceil rule (t = ceil(kappa*N_T)-th largest pooled score) and the
   held-out F1-optimal ceiling. Control: random image splits of identical sizes (20 per fold, seed 3407).
   Comparison: the fixed random calval/caltest split numbers of label_free_threshold_refine.json.
B. FAR-CONSTRAINED TRANSFER to FIgLib. On the source calval, negatives = images with no annotation of
   any class; per-image score = max smoke-class detection score (the score figlib_eval.py uses:
   max_smoke for D-Fire-trained runs, max for Pyro-SDIS-trained runs); t_FAR(f) = (1-f) quantile of the
   negative per-image max (np.quantile, same convention as figlib_eval.analyse) for f = 1/5/10 %. The
   kept-rate statistic at t_FAR is kappa_k = mean_i min(rank_i(t_FAR), k) over ALL source images with
   k = 1 (per-image max only = alarm-rate matching; exact on FIgLib from the stored per-frame max) and
   k = 3 (top-3 stored boxes; smoke-filtered for D-Fire runs, so approximate when a fire box sits in the
   top 3 - flagged). Transfer: the 4 FIgLib camera folds of figlib_ttd.json (fold construction of
   figlib_eval.analyse repeated verbatim and asserted equal); adapt on the calibration cameras' frame
   scores, evaluate on the held-out cameras with figlib_eval.frame_eval / ttd_eval (imported unchanged):
   realised pre-plume frame FAR, post-plume frame sensitivity, fraction of fires (sequences) missed,
   median TTD. Comparators on the same held-out frames: (a) the shipped t_src; (b) the labelled
   matched-FAR threshold (quantile of the calibration cameras' pre-plume frames; asserted equal to
   figlib_ttd.json). Pyro-SDIS-trained runs are also run with the pooled calval + caltest negatives
   (754 instead of 382) as a sensitivity variant. Source negatives with n_neg * f < 10 are flagged.
C. HOLM-BONFERRONI over the 44 cells of label_free_threshold_refine.json for M1c vs t_src (box F1):
   the paired image-bootstrap replicates are regenerated exactly (same seed-0 multinomial weights per
   target test, B = 1000, refine.evaluate unchanged; point F1, percentile CI and P(d <= 0) asserted equal
   to the stored values), two-sided bootstrap p = min(1, 2 min(P(d <= 0), P(d >= 0))) (the convention of
   bootstrap_ci.py), then Holm step-down at 0.05. Reported: significant gains / losses after Holm,
   Bonferroni, and the descriptive CI-excludes-0 counts.
D. TABLE-3 bootstrap means: for every in-domain <run>__to__<own test> cell in calibration_ci/ (own test
   = d_fire_test for D-Fire-family runs, pyro_sdis_caltest for Pyro-SDIS runs; plus every
   d_fire_test_dedup cell) the point estimate, bootstrap mean, percentile 95 % CI and bootstrap bias of
   LaECE_0 and D_ECE for every calibrator (identity in the table). Files mid-write by the refresh daemon
   are skipped and listed.
Writes analysis/label_free_revision.json (+ .md).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operating_points as op  # noqa: E402  read-only reuse
import label_free_threshold_refine as lf  # noqa: E402  read-only reuse (match_indexed, solve_threshold, evaluate, ci)
import camera_conformal as cc  # noqa: E402  read-only reuse (pool, camera_folds, load_run_dets)
import figlib_eval as fe  # noqa: E402  read-only reuse (frame_eval, ttd_eval)

GT, DETS = HERE / "coco_gt", HERE / "detections"
SEED = 3407
N_RANDOM = 20
FARS = (0.01, 0.05, 0.10)
KS = (1, 3)
FIGLIB_RUNS = ["v8s_dfire_s3407", "y11s_dfire_s3407", "rtdetrl_dfire_s3407",
               "v8s_pyrosdis_s3407", "y11s_pyrosdis_s3407", "rtdetrl_pyrosdis_s3407", "rtdetrl_pyrosdis_s3407_p60"]
ALPHA = 0.05
SMOKE = 2

gt_cache, det_cache = {}, {}


def gt(tag):
    if tag not in gt_cache:
        gt_cache[tag] = op.load_gt(tag)
    return gt_cache[tag]


def dets(run, tag):
    k = (run, tag)
    if k not in det_cache:
        det_cache[k] = op.load_dets(DETS / f"{run}__{tag}.bbox.json")
    return det_cache[k]


def dets_to_op(by_img):
    return {i: [(d["category_id"], d["bbox"], d["score"]) for d in v] for i, v in by_img.items()}


def subview(X, mask):
    """Unlabelled view of the detections of the images selected by mask (scores, image index, rank only)."""
    m = mask[X["img"]]
    return {"score": X["score"][m], "img": X["img"][m], "rank": X["rank"][m], "n_dets_img": X["n_dets_img"], "n_img": int(mask.sum())}


def m1c_threshold(view, kappa):
    order = np.argsort(-view["score"], kind="stable")
    return lf.solve_threshold(view, order, "M1c_kept_rate", kappa)


def m1c_ceil(view, kappa):
    """Referee's literal rule: score of the ceil(kappa * N_T)-th largest pooled target score."""
    k = int(np.ceil(kappa * view["n_img"] - 1e-9))
    if k < 1 or len(view["score"]) == 0:
        return 1.0
    s = np.sort(view["score"])[::-1]
    return float(s[min(k, len(s)) - 1])


def f1_opt_threshold(X, mask):
    m = mask[X["img"]]
    P = op.pr_curve(X["score"][m], X["tp"][m], int(X["gt_img"][mask].sum()))
    j = int(np.argmax(P[3]))
    return float(P[0][j]), float(P[3][j])


def eval_mask(X, t, mask):
    m = (X["score"] >= t) & mask[X["img"]]
    tp, nd, g = float(X["tp"][m].sum()), float(m.sum()), float(X["gt_img"][mask].sum())
    pos, neg = X["img_pos"] & mask, (~X["img_pos"]) & mask
    al = X["img_max"] >= t
    return {"f1": float(2 * tp / (nd + g)) if nd + g > 0 else 0.0, "precision": float(tp / nd) if nd > 0 else None,
            "recall": float(tp / g) if g > 0 else 0.0,
            "sensitivity": float(al[pos].mean()) if pos.any() else None, "false_alarm_rate": float(al[neg].mean()) if neg.any() else None,
            "n_images": int(mask.sum()), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "n_gt": int(g)}


def agg(rows, key, sub=None):
    v = [(r[key] if sub is None else r[key][sub]) for r in rows]
    v = [x for x in v if x is not None]
    if not v:
        return None
    return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": int(len(v))}


# =============================================================================== A
def part_a(refine):
    t0 = time.time()
    images, anns, cats = cc.pool()
    folds, sizes = cc.camera_folds(images)
    ccj = json.loads((HERE / "camera_conformal.json").read_text(encoding="utf-8"))
    assert [f["cameras"] for f in ccj["folds"]] == folds and [f["n_images"] for f in ccj["folds"]] == sizes, "fold definition differs from camera_conformal.json"
    img_ids = [im["id"] for im in images]
    gt_boxes = {}
    for a in anns:
        gt_boxes.setdefault(a["image_id"], []).append((a["category_id"], a["bbox"]))
    cam_arr = np.array([im["camera"] for im in images])
    test_masks = [np.isin(cam_arr, f) for f in folds]
    cells = sorted(c for c, e in refine["cells"].items() if e["target_test"] == "pyro_sdis_caltest")
    METHODS = ["M0_t_src", "M1c_kept_rate", "M1c_ceil_rule", "ORC_cal_cameras", "TOPT_heldout"]
    out = {"n_images": len(images), "n_positive": len({a["image_id"] for a in anns}), "n_cameras": len(set(cam_arr)),
           "folds": [{"fold": i, "test_cameras": f, "n_test_images": s} for i, (f, s) in enumerate(zip(folds, sizes))],
           "random_control": {"n_per_fold": N_RANDOM, "seed": SEED}, "methods": METHODS, "cells": {}}
    for name in cells:
        tc = time.time()
        e = refine["cells"][name]
        run, src = e["run"], e["source_calval"]
        cls_s, cls_t = set(e["classes_source"]), set(e["classes_evaluated"])
        s_ids, s_gt, _ = gt(src)
        S = lf.match_indexed(s_ids, s_gt, dets(run, src), cls_s)
        Sc = op.pr_curve(S["score"], S["tp"], S["n_gt"])
        t_src = float(Sc[0][int(np.argmax(Sc[3]))])
        kappa = lf.source_statistic(lf.unlabelled_view(S), t_src, "M1c_kept_rate")
        assert abs(t_src - e["source"]["t_src"]) < 1e-9, (name, t_src, e["source"]["t_src"])
        assert abs(kappa - e["source"]["statistics_at_t_src"]["M1c_kept_rate"]) < 1e-9, (name, kappa)
        pd = cc.load_run_dets(run, images)
        assert pd is not None, f"{run}: pyro_sdis detections missing"
        X = lf.match_indexed(img_ids, gt_boxes, dets_to_op(pd), cls_t)
        per_fold = []
        for fi, te in enumerate(test_masks):
            ca = ~te
            v = subview(X, ca)
            thr = {"M0_t_src": t_src, "M1c_kept_rate": m1c_threshold(v, kappa), "M1c_ceil_rule": m1c_ceil(v, kappa)}
            thr["ORC_cal_cameras"], f1_orc_cal = f1_opt_threshold(X, ca)
            thr["TOPT_heldout"], _ = f1_opt_threshold(X, te)
            row = {"fold": fi, "n_cal_images": int(ca.sum()), "n_test_images": int(te.sum()), "thresholds": thr,
                   "kept_per_image_on_cal_at_M1c": float(np.sum(v["score"] >= thr["M1c_kept_rate"]) / v["n_img"]),
                   "f1_oracle_on_cal_cameras": f1_orc_cal,
                   "heldout": {m: eval_mask(X, t, te) for m, t in thr.items()}}
            row["regret"] = {m: row["heldout"]["ORC_cal_cameras"]["f1"] - row["heldout"][m]["f1"] for m in METHODS}
            per_fold.append(row)
        # random image-split control with identical sizes (exchangeable reference)
        rng = np.random.default_rng(SEED)
        rnd = []
        for fi, te in enumerate(test_masks):
            n_te = int(te.sum())
            for _ in range(N_RANDOM):
                perm = rng.permutation(len(img_ids))
                te_r = np.zeros(len(img_ids), dtype=bool); te_r[perm[:n_te]] = True
                ca_r = ~te_r
                v = subview(X, ca_r)
                thr = {"M0_t_src": t_src, "M1c_kept_rate": m1c_threshold(v, kappa), "M1c_ceil_rule": m1c_ceil(v, kappa)}
                thr["ORC_cal_cameras"], _ = f1_opt_threshold(X, ca_r)
                thr["TOPT_heldout"], _ = f1_opt_threshold(X, te_r)
                h = {m: eval_mask(X, t, te_r) for m, t in thr.items()}
                rnd.append({"heldout": h, "regret": {m: h["ORC_cal_cameras"]["f1"] - h[m]["f1"] for m in METHODS}, "thresholds": thr})
        summ = {}
        for m in METHODS:
            summ[m] = {"camera_disjoint": {"threshold": agg(per_fold, "thresholds", m), "f1": agg([r["heldout"] for r in per_fold], m, "f1"),
                                           "regret": agg(per_fold, "regret", m),
                                           "sensitivity": agg([r["heldout"] for r in per_fold], m, "sensitivity"),
                                           "false_alarm_rate": agg([r["heldout"] for r in per_fold], m, "false_alarm_rate")},
                       "random_control": {"threshold": agg(rnd, "thresholds", m), "f1": agg([r["heldout"] for r in rnd], m, "f1"),
                                          "regret": agg(rnd, "regret", m),
                                          "sensitivity": agg([r["heldout"] for r in rnd], m, "sensitivity"),
                                          "false_alarm_rate": agg([r["heldout"] for r in rnd], m, "false_alarm_rate")}}
        ref = {m: {k: e["methods"][m][k] for k in ("threshold", "f1", "regret_vs_oracle", "sensitivity", "false_alarm_rate")}
               for m in ("M0_t_src", "M1c_kept_rate", "ORC_target_calval")}
        d_ceil = max(abs(r["thresholds"]["M1c_ceil_rule"] - r["thresholds"]["M1c_kept_rate"]) for r in per_fold)
        out["cells"][name] = {"run": run, "family": e["family"], "source_calval": src, "classes_source": sorted(cls_s), "classes_evaluated": sorted(cls_t),
                              "t_src": t_src, "kappa_kept_per_image_at_t_src": kappa,
                              "n_pooled_dets": int(len(X["score"])), "n_pooled_gt": X["n_gt"],
                              "per_fold": per_fold, "summary": summ,
                              "ceil_rule_vs_refine_solver_max_abs_threshold_diff": d_ceil,
                              "refine_random_split_reference": ref, "seconds": round(time.time() - tc, 1)}
        s = summ
        print(f"[A] {name}: kappa {kappa:.3f} | held-out F1 t_src {s['M0_t_src']['camera_disjoint']['f1']['mean']:.3f} "
              f"M1c {s['M1c_kept_rate']['camera_disjoint']['f1']['mean']:.3f} (worst {s['M1c_kept_rate']['camera_disjoint']['f1']['min']:.3f}) "
              f"ORC {s['ORC_cal_cameras']['camera_disjoint']['f1']['mean']:.3f} | regret M1c {s['M1c_kept_rate']['camera_disjoint']['regret']['mean']:.3f} "
              f"(worst {s['M1c_kept_rate']['camera_disjoint']['regret']['max']:.3f}) vs refine {ref['M1c_kept_rate']['regret_vs_oracle']:.3f} | {time.time() - tc:.0f}s", flush=True)
    # across cells
    C = out["cells"]
    def across(sel, m, kind, key, stat):
        v = [C[c]["summary"][m][kind][key][stat] for c in sel if C[c]["summary"][m][kind][key] is not None]
        return float(np.mean(v)) if v else None
    groups = {"all": cells, "yolo": [c for c in cells if C[c]["family"] == "yolo"], "rtdetr": [c for c in cells if C[c]["family"] == "rtdetr"]}
    out["summary"] = {}
    for g, sel in groups.items():
        if not sel:
            continue
        out["summary"][g] = {"n_cells": len(sel)}
        for m in METHODS:
            out["summary"][g][m] = {"camera_disjoint": {"regret_foldmean_mean": across(sel, m, "camera_disjoint", "regret", "mean"),
                                                        "regret_worstfold_mean": across(sel, m, "camera_disjoint", "regret", "max"),
                                                        "regret_worstfold_max": max(C[c]["summary"][m]["camera_disjoint"]["regret"]["max"] for c in sel),
                                                        "f1_foldmean_mean": across(sel, m, "camera_disjoint", "f1", "mean"),
                                                        "sensitivity_foldmean_mean": across(sel, m, "camera_disjoint", "sensitivity", "mean"),
                                                        "false_alarm_rate_foldmean_mean": across(sel, m, "camera_disjoint", "false_alarm_rate", "mean")},
                                    "random_control": {"regret_mean": across(sel, m, "random_control", "regret", "mean"),
                                                       "f1_mean": across(sel, m, "random_control", "f1", "mean")}}
            if m in ("M0_t_src", "M1c_kept_rate"):
                rm = "M0_t_src" if m == "M0_t_src" else "M1c_kept_rate"
                out["summary"][g][m]["refine_random_split"] = {"regret_mean": float(np.mean([C[c]["refine_random_split_reference"][rm]["regret_vs_oracle"] for c in sel])),
                                                              "f1_mean": float(np.mean([C[c]["refine_random_split_reference"][rm]["f1"] for c in sel]))}
        out["summary"][g]["n_cells_M1c_beats_t_src_every_fold"] = int(sum(1 for c in sel if all(r["heldout"]["M1c_kept_rate"]["f1"] > r["heldout"]["M0_t_src"]["f1"] for r in C[c]["per_fold"])))
        out["summary"][g]["n_cells_M1c_beats_t_src_foldmean"] = int(sum(1 for c in sel if C[c]["summary"]["M1c_kept_rate"]["camera_disjoint"]["f1"]["mean"] > C[c]["summary"]["M0_t_src"]["camera_disjoint"]["f1"]["mean"]))
    out["seconds"] = round(time.time() - t0, 1)
    return out


# =============================================================================== B
def figlib_folds(frames):
    """Verbatim fold construction of figlib_eval.analyse (largest camera first, to the smallest fold)."""
    counts = defaultdict(int)
    for f in frames:
        counts[f["camera"]] += 1
    folds, sizes = [[] for _ in range(4)], [0] * 4
    for cam, n in sorted(counts.items(), key=lambda x: -x[1]):
        i = int(np.argmin(sizes)); folds[i].append(cam); sizes[i] += n
    return folds, sizes


def source_views(run, tags):
    """Per-image smoke-class score lists of the source calval (optionally pooled tags); negatives = no annotation of any class."""
    per_img, neg_max = {}, []
    for tag in tags:
        ids, boxes, _ = gt(tag)
        d = dets(run, tag)
        for i in ids:
            ss = sorted([s for c, _, s in d.get(i, []) if c == SMOKE], reverse=True)
            per_img[(tag, i)] = ss
            if not boxes.get(i):
                neg_max.append(ss[0] if ss else 0.0)
    return per_img, np.array(neg_max, dtype=float)


def kappa_k(per_img, t, k):
    return float(sum(min(sum(1 for s in ss if s >= t), k) for ss in per_img.values()) / len(per_img))


def part_b():
    t0 = time.time()
    g = json.loads((HERE / "coco_gt" / "figlib_all.json").read_text(encoding="utf-8"))
    frames = g["images"]
    ttd = json.loads((HERE / "figlib_ttd.json").read_text(encoding="utf-8"))
    opj = json.loads((HERE / "operating_points.json").read_text(encoding="utf-8"))["cells"]
    folds, sizes = figlib_folds(frames)
    assert [f["cameras"] for f in ttd["folds"]] == folds and [f["n_frames"] for f in ttd["folds"]] == sizes, "FIgLib folds differ from figlib_ttd.json"
    cam_arr = np.array([f["camera"] for f in frames])
    test_masks = [np.isin(cam_arr, fo) for fo in folds]
    fold_frames = [([f for f, m in zip(frames, te) if m], [f for f, m in zip(frames, te) if not m]) for te in test_masks]
    n_pre = sum(1 for f in frames if not f["smoke_visible"]); n_post = len(frames) - n_pre
    out = {"n_frames": len(frames), "n_pre_plume_frames": n_pre, "n_post_plume_frames": n_post, "n_sequences": ttd["n_sequences"],
           "folds": [{"fold": i, "n_cameras": len(f), "n_frames": s} for i, (f, s) in enumerate(zip(folds, sizes))],
           "fars": list(FARS), "ks": list(KS), "runs": {}}
    EVAL_KEYS = ("false_alarm_rate", "sensitivity", "missed_frac", "ttd_min_median", "detected_within_10min_frac", "sequence_false_alarm_frac")
    for run in FIGLIB_RUNS:
        tr = time.time()
        raw = json.loads((HERE / "figlib_scores" / f"{run}.json").read_text(encoding="utf-8"))
        assert len(raw) >= len(frames), f"{run}: {len(raw)}/{len(frames)} frames scored"
        smoke_only = "_dfire_" in run
        key = "max_smoke" if smoke_only else "max"
        score = {k: v[key] for k, v in raw.items()}
        assert key == ttd["runs"][run]["score_used"]
        # FIgLib unlabelled views (k = 1 exact per-frame max; k = 3 from the stored top-3 boxes)
        views, n_top3_truncated = {}, 0
        for k in KS:
            S_, I_, R_ = [], [], []
            for i, f in enumerate(frames):
                v = raw[str(f["id"])]
                if k == 1:
                    ss = [v[key]] if v[key] > 0 else []
                else:
                    ss = sorted([s for s, c, _ in v["top3"] if (c == "smoke" or not smoke_only)], reverse=True)[:k]
                    if smoke_only and len(ss) < min(k, v["n"]) and any(c != "smoke" for _, c, _ in v["top3"]):
                        n_top3_truncated += 1
                for r, s in enumerate(ss, 1):
                    S_.append(s); I_.append(i); R_.append(r)
            I_ = np.array(I_, dtype=np.int64)
            views[k] = {"score": np.array(S_, dtype=float), "img": I_, "rank": np.array(R_, dtype=np.int64),
                        "n_img": len(frames), "n_dets_img": np.bincount(I_, minlength=len(frames))}
        t_src = ttd["runs"][run]["t_src"]
        assert abs(t_src - next(o["source"]["t_f1opt"] for c, o in opj.items() if c.startswith(run + "__to__"))) < 1e-12
        src_variants = [("d_fire_calval",)] if smoke_only else [("pyro_sdis_calval",), ("pyro_sdis_calval", "pyro_sdis_caltest")]
        rr = {"score_used": key, "t_src": t_src, "n_frames_top3_smoke_filter_truncated": n_top3_truncated, "source_variants": {}}
        for tags in src_variants:
            per_img, neg_max = source_views(run, tags)
            vname = "+".join(tags)
            vv = {"source_tags": list(tags), "n_source_images": len(per_img), "n_source_negatives": int(len(neg_max)), "by_far": {}}
            for far in FARS:
                t_far = float(np.quantile(neg_max, 1 - far))
                kap = {k: kappa_k(per_img, t_far, k) for k in KS}
                per_fold = []
                for fi, (te_fr, ca_fr) in enumerate(fold_frames):
                    ca = ~test_masks[fi]
                    neg_cal = [score[str(f["id"])] for f in ca_fr if not f["smoke_visible"]]
                    t_lab = float(np.quantile(neg_cal, 1 - far))
                    ref = ttd["runs"][run]["at_matched_far"][str(far)]["camera_disjoint_folds"][fi]
                    assert ref["fold"] == fi and abs(t_lab - ref["t"]) < 1e-9, (run, far, fi, t_lab, ref["t"])
                    thr = {"t_src": t_src, "labelled_matched_far": t_lab}
                    for k in KS:
                        thr[f"kept_rate_k{k}"] = m1c_threshold(subview(views[k], ca), kap[k])
                    h = {m: {**fe.frame_eval(te_fr, score, t), **fe.ttd_eval(te_fr, score, t)} for m, t in thr.items()}
                    # reproduction of figlib_ttd's held-out numbers for the labelled threshold
                    for kk in ("sensitivity", "false_alarm_rate", "missed_frac"):
                        assert abs(h["labelled_matched_far"][kk] - ref[kk]) < 1e-12, (run, far, fi, kk)
                    per_fold.append({"fold": fi, "thresholds": thr, "heldout": h})
                pooled = {}
                for k in KS:
                    t = m1c_threshold(views[k], kap[k])
                    pooled[f"kept_rate_k{k}"] = {"t": t, **fe.frame_eval(frames, score, t), **fe.ttd_eval(frames, score, t)}
                pooled["labelled_matched_far"] = {"t": ttd["runs"][run]["at_matched_far"][str(far)]["pooled"]["t"],
                                                  **{kk: ttd["runs"][run]["at_matched_far"][str(far)]["pooled"][kk] for kk in EVAL_KEYS}}
                pooled["t_src"] = {"t": t_src, **{kk: ttd["runs"][run]["at_t_src"][kk] for kk in EVAL_KEYS}}
                summ = {}
                for m in per_fold[0]["thresholds"]:
                    summ[m] = {"threshold": agg(per_fold, "thresholds", m)}
                    for kk in EVAL_KEYS:
                        summ[m][kk] = agg([r["heldout"] for r in per_fold], m, kk)
                vv["by_far"][str(far)] = {"t_far_source": t_far, "realised_source_far_at_t_far": float(np.mean(neg_max >= t_far)),
                                         "n_effective_source_negatives_above": float(len(neg_max) * far),
                                         "few_negatives_flag": bool(len(neg_max) * far < 10),
                                         "kappa_at_t_far": {str(k): v for k, v in kap.items()}, "per_fold": per_fold, "camera_disjoint_summary": summ, "pooled_all_frames": pooled}
                s = summ
                print(f"[B] {run} [{vname}] FAR {far:.2f}: t_far {t_far:.3f} (n_neg {len(neg_max)}, kappa1 {kap[1]:.3f}) | held-out FAR: "
                      f"t_src {s['t_src']['false_alarm_rate']['mean']:.3f} k1 {s['kept_rate_k1']['false_alarm_rate']['mean']:.3f} "
                      f"(worst {s['kept_rate_k1']['false_alarm_rate']['max']:.3f}) k3 {s['kept_rate_k3']['false_alarm_rate']['mean']:.3f} "
                      f"lab {s['labelled_matched_far']['false_alarm_rate']['mean']:.3f} | sens k1 {s['kept_rate_k1']['sensitivity']['mean']:.3f} "
                      f"lab {s['labelled_matched_far']['sensitivity']['mean']:.3f} | missed k1 {s['kept_rate_k1']['missed_frac']['mean']:.3f} "
                      f"lab {s['labelled_matched_far']['missed_frac']['mean']:.3f}", flush=True)
            rr["source_variants"][vname] = vv
        rr["seconds"] = round(time.time() - tr, 1)
        out["runs"][run] = rr
    out["seconds"] = round(time.time() - t0, 1)
    return out


# =============================================================================== C
def holm(p, alpha):
    p = np.asarray(p, dtype=float); m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, p[idx] * (m - i))
        adj[idx] = min(1.0, running)
    return adj, adj <= alpha


def part_c(refine, B_override=None):
    t0 = time.time()
    B_ = refine["bootstrap"]["B"] if B_override is None else B_override
    seed = refine["bootstrap"]["seed"]
    W_cache = {}

    def weights(tag, n_img):
        if tag not in W_cache:
            rng = np.random.default_rng(seed)
            W_cache[tag] = rng.multinomial(n_img, np.full(n_img, 1.0 / n_img), size=B_).astype(float)
        assert W_cache[tag].shape == (B_, n_img)
        return W_cache[tag]

    rows = []
    for name, e in refine["cells"].items():
        tc = time.time()
        run, tgt, cls_t = e["run"], e["target_test"], set(e["classes_evaluated"])
        x_ids, x_gt, _ = gt(tgt)
        X = lf.match_indexed(x_ids, x_gt, dets(run, tgt), cls_t)
        W = weights(tgt, X["n_img"])
        t0_, t1_ = e["methods"]["M0_t_src"]["threshold"], e["methods"]["M1c_kept_rate"]["threshold"]
        p0, r0 = lf.evaluate(X, t0_, W)
        p1, r1 = lf.evaluate(X, t1_, W)
        assert abs(p0["f1"] - e["methods"]["M0_t_src"]["f1"]) < 1e-9 and abs(p1["f1"] - e["methods"]["M1c_kept_rate"]["f1"]) < 1e-9, name
        d = r1["f1"] - r0["f1"]
        st = e["methods"]["M1c_kept_rate"]["diff_vs_t_src"]
        ci_ = lf.ci(d)
        if B_override is None:
            assert abs(ci_[0] - st["ci95"][0]) < 1e-9 and abs(ci_[1] - st["ci95"][1]) < 1e-9, (name, ci_, st["ci95"])
            assert abs(float(np.mean(d <= 0)) - st["p_boot_le_0"]) < 1e-12, name
        p_le, p_ge = float(np.mean(d <= 0)), float(np.mean(d >= 0))
        rows.append({"cell": name, "family": e["family"], "target": e["target"], "point_diff_f1": float(p1["f1"] - p0["f1"]),
                     "ci95": ci_, "p_le_0": p_le, "p_ge_0": p_ge, "n_replicates_exactly_0": int(np.sum(d == 0)),
                     "p_two_sided_boot": float(min(1.0, 2 * min(p_le, p_ge))), "seconds": round(time.time() - tc, 1)})
        print(f"[C] {name}: diff {rows[-1]['point_diff_f1']:+.3f} CI [{ci_[0]:+.3f}, {ci_[1]:+.3f}] p2 {rows[-1]['p_two_sided_boot']:.3f} | {time.time() - tc:.0f}s", flush=True)
    p = np.array([r["p_two_sided_boot"] for r in rows])
    diff = np.array([r["point_diff_f1"] for r in rows])
    adj, sig = holm(p, ALPHA)
    bonf = p <= ALPHA / len(p)
    for r, a, s, b in zip(rows, adj, sig, bonf):
        r["p_holm_adjusted"] = float(a); r["significant_holm_0.05"] = bool(s); r["significant_bonferroni_0.05"] = bool(b)
        r["significant_unadjusted_0.05"] = bool(r["p_two_sided_boot"] <= ALPHA)
    ci_lo = np.array([r["ci95"][0] for r in rows]); ci_hi = np.array([r["ci95"][1] for r in rows])
    fam = np.array([r["family"] for r in rows])
    def counts(mask):
        return {"n": int(mask.sum()), "holm_gain": int(np.sum(sig & (diff > 0) & mask)), "holm_loss": int(np.sum(sig & (diff < 0) & mask)),
                "bonferroni_gain": int(np.sum(bonf & (diff > 0) & mask)), "bonferroni_loss": int(np.sum(bonf & (diff < 0) & mask)),
                "unadjusted_gain": int(np.sum((p <= ALPHA) & (diff > 0) & mask)), "unadjusted_loss": int(np.sum((p <= ALPHA) & (diff < 0) & mask)),
                "ci_excludes_0_gain": int(np.sum((ci_lo > 0) & mask)), "ci_excludes_0_loss": int(np.sum((ci_hi < 0) & mask)),
                "point_gain": int(np.sum((diff > 0) & mask)), "point_loss": int(np.sum((diff < 0) & mask))}
    out = {"B": B_, "seed": seed, "alpha": ALPHA, "n_cells": len(rows), "resolution_note": f"bootstrap p has resolution 1/B = {1 / B_:g}; p = 0 means no replicate on the opposite side (true p < ~{1 / B_:g}); Holm's strictest level is alpha/m = {ALPHA / len(rows):.5f}",
           "counts": {"all": counts(np.ones(len(rows), bool)), "yolo": counts(fam == "yolo"), "rtdetr": counts(fam == "rtdetr")},
           "n_cells_p_two_sided_equal_0": int(np.sum(p == 0)), "cells": rows, "seconds": round(time.time() - t0, 1)}
    return out


# =============================================================================== D
def part_d():
    t0 = time.time()
    rows, skipped = {}, []
    for p in sorted((HERE / "calibration_ci").glob("*.json")):
        cell = p.stem
        if "__repair" in cell or "__to__" not in cell:
            continue
        run, tgt = cell.split("__to__")
        train = run.split("_")[1]
        own = "d_fire_test" if train in ("dfire", "dfiredd", "dfiresub") else ("pyro_sdis_caltest" if train == "pyrosdis" else None)
        wanted = {own} | ({"d_fire_test_dedup"} if own == "d_fire_test" else set())
        if tgt not in wanted:
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception as ex:  # file mid-write by the refresh daemon
            skipped.append({"cell": cell, "why": f"unreadable ({type(ex).__name__})"}); continue
        rows[cell] = {"run": run, "test": tgt, "train_set": train, "B": j.get("B"), "seed": j.get("seed"), "unit": j.get("unit"), "calibrators": {}}
        for cal, res in j["results"].items():
            rows[cell]["calibrators"][cal] = {m: {"point": res[m]["point"], "boot_mean": res[m]["boot_mean"], "ci95": res[m]["ci95"],
                                                  "boot_bias": float(res[m]["boot_mean"] - res[m]["point"]),
                                                  "point_inside_ci": bool(res[m]["ci95"][0] <= res[m]["point"] <= res[m]["ci95"][1])}
                                              for m in ("LaECE_0", "D_ECE")}
    bias = [r["calibrators"]["identity"]["LaECE_0"]["boot_bias"] for r in rows.values() if "identity" in r["calibrators"]]
    return {"n_cells": len(rows), "skipped": skipped, "cells": rows,
            "identity_LaECE_0_boot_bias": {"mean": float(np.mean(bias)) if bias else None, "max": float(np.max(bias)) if bias else None,
                                           "n_positive": int(sum(1 for b in bias if b > 0)), "n": len(bias)},
            "seconds": round(time.time() - t0, 1)}


# =============================================================================== md
def f3(x):
    return "" if x is None else f"{x:.3f}"


def mm(a, stat="mean"):
    return "" if a is None else f3(a[stat])


def write_md(out):
    L = ["# Referee-revision analyses for the label-free operating-point transfer (machine-written by label_free_revision.py)", "",
         f"Created {out['created']}; total {out['seconds_total']} s CPU. Parts: A camera-disjoint M1c (Pyro-SDIS targets), "
         "B FAR-constrained transfer to FIgLib, C Holm correction, D Table-3 bootstrap means.", ""]
    A = out["A"]
    L += [f"## A. Camera-disjoint kept-rate rule on Pyro-SDIS ({A['n_images']} pooled images, {A['n_cameras']} cameras, 4 folds)", "",
          "Held-out-camera box F1 (fold mean / worst fold), regret vs the oracle fitted on the calibration cameras with labels, "
          "image-level sensitivity / FAR (fold mean); 'refine' = fixed random calval/caltest split of label_free_threshold_refine.json; "
          "'rnd' = random image splits of identical sizes (20 per fold).", "",
          "| cell | kappa | F1 t_src fold-mean / worst | F1 M1c fold-mean / worst | F1 ORC fold-mean / worst | regret M1c mean / worst | regret t_src mean / worst | refine: F1 t_src / M1c / ORC, regret M1c | rnd regret M1c mean / max | sens M1c / t_src / ORC | FAR M1c / t_src / ORC | t_M1c folds |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, e in A["cells"].items():
        s, r = e["summary"], e["refine_random_split_reference"]
        cd = lambda m, k, st="mean": s[m]["camera_disjoint"][k][st]
        L.append(f"| {c} | {e['kappa_kept_per_image_at_t_src']:.3f} | {f3(cd('M0_t_src','f1'))} / {f3(cd('M0_t_src','f1','min'))} | "
                 f"{f3(cd('M1c_kept_rate','f1'))} / {f3(cd('M1c_kept_rate','f1','min'))} | {f3(cd('ORC_cal_cameras','f1'))} / {f3(cd('ORC_cal_cameras','f1','min'))} | "
                 f"{f3(cd('M1c_kept_rate','regret'))} / {f3(cd('M1c_kept_rate','regret','max'))} | {f3(cd('M0_t_src','regret'))} / {f3(cd('M0_t_src','regret','max'))} | "
                 f"{f3(r['M0_t_src']['f1'])} / {f3(r['M1c_kept_rate']['f1'])} / {f3(r['ORC_target_calval']['f1'])}, {f3(r['M1c_kept_rate']['regret_vs_oracle'])} | "
                 f"{mm(s['M1c_kept_rate']['random_control']['regret'])} / {mm(s['M1c_kept_rate']['random_control']['regret'], 'max')} | "
                 f"{f3(cd('M1c_kept_rate','sensitivity'))} / {f3(cd('M0_t_src','sensitivity'))} / {f3(cd('ORC_cal_cameras','sensitivity'))} | "
                 f"{f3(cd('M1c_kept_rate','false_alarm_rate'))} / {f3(cd('M0_t_src','false_alarm_rate'))} / {f3(cd('ORC_cal_cameras','false_alarm_rate'))} | "
                 f"{' '.join(f3(pf['thresholds']['M1c_kept_rate']) for pf in e['per_fold'])} |")
    L += ["", "| group | n | regret M1c: camera-disjoint fold-mean | worst-fold mean | worst-fold max | rnd control | refine random split | regret t_src: camera-disjoint | refine | M1c beats t_src (fold-mean) | every fold |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, s in A["summary"].items():
        L.append(f"| {g} | {s['n_cells']} | {f3(s['M1c_kept_rate']['camera_disjoint']['regret_foldmean_mean'])} | {f3(s['M1c_kept_rate']['camera_disjoint']['regret_worstfold_mean'])} | "
                 f"{f3(s['M1c_kept_rate']['camera_disjoint']['regret_worstfold_max'])} | {f3(s['M1c_kept_rate']['random_control']['regret_mean'])} | {f3(s['M1c_kept_rate']['refine_random_split']['regret_mean'])} | "
                 f"{f3(s['M0_t_src']['camera_disjoint']['regret_foldmean_mean'])} | {f3(s['M0_t_src']['refine_random_split']['regret_mean'])} | {s['n_cells_M1c_beats_t_src_foldmean']} | {s['n_cells_M1c_beats_t_src_every_fold']} |")
    Bp = out["B"]
    L += ["", f"## B. FAR-constrained transfer to FIgLib ({Bp['n_frames']} frames, {Bp['n_pre_plume_frames']} pre-plume, {Bp['n_sequences']} sequences, 4 camera folds)", "",
          "Held-out-camera fold means (worst fold in brackets: max FAR, min sensitivity, max missed). k1 = kept-rate rule on the per-frame max "
          "(alarm-rate matching, exact); k3 = kept-rate on the stored top-3 boxes; lab = labelled matched-FAR threshold of figlib_ttd.json "
          "(calibration cameras' pre-plume frames); t_src = shipped threshold.", "",
          "| run | source negatives | FAR target | t_far (src) | src FAR realised | kappa1 | t: k1 / k3 / lab (fold-mean) | FAR: t_src / k1 / k3 / lab | sens: t_src / k1 / k3 / lab | missed: t_src / k1 / k3 / lab | median TTD min: t_src / k1 / k3 / lab |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, rr in Bp["runs"].items():
        for vname, vv in rr["source_variants"].items():
            for far, b in vv["by_far"].items():
                s = b["camera_disjoint_summary"]
                def q(kk, st="mean", ms=("t_src", "kept_rate_k1", "kept_rate_k3", "labelled_matched_far")):
                    return " / ".join(mm(s[m][kk], st) for m in ms)
                flag = " (few)" if b["few_negatives_flag"] else ""
                L.append(f"| {run} | {vname} n={vv['n_source_negatives']}{flag} | {far} | {b['t_far_source']:.3f} | {b['realised_source_far_at_t_far']:.3f} | {b['kappa_at_t_far']['1']:.3f} | "
                         f"{q('threshold', 'mean', ('kept_rate_k1', 'kept_rate_k3', 'labelled_matched_far'))} | "
                         f"{q('false_alarm_rate')} [{q('false_alarm_rate', 'max')}] | {q('sensitivity')} [{q('sensitivity', 'min')}] | "
                         f"{q('missed_frac')} [{q('missed_frac', 'max')}] | {q('ttd_min_median')} |")
    Cp = out["C"]
    L += ["", f"## C. Holm-Bonferroni over {Cp['n_cells']} cells, M1c vs t_src (box F1, paired image bootstrap B = {Cp['B']}, seed {Cp['seed']})", "",
          f"{Cp['resolution_note']}. Cells with two-sided p = 0: {Cp['n_cells_p_two_sided_equal_0']}.", "",
          "| group | n | Holm gain / loss | Bonferroni gain / loss | unadjusted gain / loss | CI excludes 0 gain / loss | point gain / loss |", "|---|---|---|---|---|---|---|"]
    for g, c in Cp["counts"].items():
        L.append(f"| {g} | {c['n']} | {c['holm_gain']} / {c['holm_loss']} | {c['bonferroni_gain']} / {c['bonferroni_loss']} | {c['unadjusted_gain']} / {c['unadjusted_loss']} | "
                 f"{c['ci_excludes_0_gain']} / {c['ci_excludes_0_loss']} | {c['point_gain']} / {c['point_loss']} |")
    L += ["", "| cell | diff F1 | CI95 | p two-sided | p Holm | sig Holm |", "|---|---|---|---|---|---|"]
    for r in sorted(Cp["cells"], key=lambda r: r["p_two_sided_boot"]):
        L.append(f"| {r['cell']} | {r['point_diff_f1']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p_two_sided_boot']:.3f} | {r['p_holm_adjusted']:.3f} | {'yes' if r['significant_holm_0.05'] else ''} |")
    Dp = out["D"]
    L += ["", f"## D. Table-3 in-domain cells: identity LaECE_0 / D_ECE point, bootstrap mean, percentile 95 % CI ({Dp['n_cells']} cells from calibration_ci/; skipped {len(Dp['skipped'])})", "",
          "| cell | B | LaECE_0 point | boot mean | CI95 | bias | D_ECE point | boot mean | CI95 | bias |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in Dp["cells"].items():
        i = r["calibrators"].get("identity")
        if i is None:
            continue
        L.append(f"| {c} | {r['B']} | {f3(i['LaECE_0']['point'])} | {f3(i['LaECE_0']['boot_mean'])} | [{f3(i['LaECE_0']['ci95'][0])}, {f3(i['LaECE_0']['ci95'][1])}] | {i['LaECE_0']['boot_bias']:+.4f} | "
                 f"{f3(i['D_ECE']['point'])} | {f3(i['D_ECE']['boot_mean'])} | [{f3(i['D_ECE']['ci95'][0])}, {f3(i['D_ECE']['ci95'][1])}] | {i['D_ECE']['boot_bias']:+.4f} |")
    bb = Dp["identity_LaECE_0_boot_bias"]
    L.append(f"\nIdentity LaECE_0 bootstrap bias (boot mean - point): mean {f3(bb['mean'])}, max {f3(bb['max'])}, positive in {bb['n_positive']}/{bb['n']} cells.")
    (HERE / "label_free_revision.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ABCD")
    ap.add_argument("--B", type=int, default=None, help="debug only: override bootstrap B (disables the reproduction asserts of part C)")
    ap.add_argument("--md-only", action="store_true", help="rewrite label_free_revision.md from the existing JSON (no recomputation)")
    args = ap.parse_args()
    if args.md_only:
        out = json.loads((HERE / "label_free_revision.json").read_text(encoding="utf-8"))
        L = write_md(out)
        print("\n".join(L[:12]))
        print("markdown rewritten from", HERE / "label_free_revision.json")
        return
    t_start = time.time()
    refine = json.loads((HERE / "label_free_threshold_refine.json").read_text(encoding="utf-8"))
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "protocol": __doc__, "seed": SEED,
           "inputs": {"refine": str(HERE / "label_free_threshold_refine.json"), "camera_conformal": str(HERE / "camera_conformal.json"),
                      "figlib_ttd": str(HERE / "figlib_ttd.json"), "figlib_scores": str(HERE / "figlib_scores"), "calibration_ci": str(HERE / "calibration_ci"),
                      "refine_created": refine["created"]}}
    if "A" in args.parts:
        out["A"] = part_a(refine)
    if "B" in args.parts:
        out["B"] = part_b()
    if "C" in args.parts:
        out["C"] = part_c(refine, args.B)
    if "D" in args.parts:
        out["D"] = part_d()
    out["seconds_total"] = round(time.time() - t_start, 1)
    (HERE / "label_free_revision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    if args.parts == "ABCD":
        L = write_md(out)
        print("\n".join(L[:12]))
    print("artefact ->", HERE / "label_free_revision.json", f"({out['seconds_total']} s)")


if __name__ == "__main__":
    main()
