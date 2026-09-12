"""Image-level bootstrap confidence intervals for the calibration cells.

For one cell (as written by run_calibration.py) this refits each calibrator
exactly as run_calibration does, evaluates it with the toolbox (asserting the
point estimate reproduces the cell artefact to 1e-6), then bootstraps the
TEST images (B resamples with replacement, seed 3407) and recomputes LaECE_0
(25 bins, tau=0, class-mean) and D-ECE (10 bins, tau=0.5, class-agnostic)
from the toolbox's own per-image evaluation records (evalImgs), so the
resampled statistic is the toolbox statistic, not a re-implementation
(the full-sample recomputation is asserted equal to the toolbox value).

Also reports PAIRED bootstrap CIs of (calibrator - identity): the same image
resamples are used for every calibrator, which is what the "post-hoc harms /
helps" claims need.
Writes analysis/calibration_ci/<cell>.json.
Usage: python analysis/bootstrap_ci.py --cell <name> [--B 1000]
       python analysis/bootstrap_ci.py --all      # every cell lacking a CI
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CAL = HERE / "calibration"
OUT = HERE / "calibration_ci"
CALIBRATORS = ["identity", "temperature_scaling", "platt_scaling", "isotonic_regression"]
SEED = 3407


def _q(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def per_image_records(test_gt, dets, bin_count, tau, is_dece, max_dets=100):
    """Run toolbox evaluate() once; return per-image, per-class arrays plus
    the toolbox's own point estimate for the reproduction assert."""
    from detection_calibration.coco_calibration import CalibrationCOCO
    from pycocotools.coco import COCO
    ev = CalibrationCOCO(test_gt, test_gt, "bbox", bin_count, tau, is_dece, False, max_dets)
    ev.cocoDt = _q(COCO(test_gt).loadRes, dets)
    _q(ev.evaluate)
    p = ev._paramsEval
    I0, K = len(p.imgIds), len(p.catIds)
    recs = []  # recs[k][i] = (scores, tps, fps, iou*tp, npig) or None
    for k in range(K):
        row = []
        for i in range(I0):
            e = ev.evalImgs[k * I0 + i]  # A0 == 1 (single area range)
            if e is None:
                row.append(None)
                continue
            sc = np.asarray(e["dtScores"][:max_dets], dtype=float)
            dtm = np.asarray(e["dtMatches"][:, :max_dets])[0]
            dtIg = np.asarray(e["dtIgnore"][:, :max_dets])[0].astype(bool)
            dtIoU = np.asarray(e["dtIoUs"][:, :max_dets])[0]
            tps = np.logical_and(dtm, ~dtIg)
            fps = np.logical_and(~dtm.astype(bool), ~dtIg)
            npig = int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
            row.append((sc, tps, fps, dtIoU * tps, npig))
        recs.append(row)
    _q(ev.prepare_input)
    _q(ev.compute_single_errors)
    point = float(ev.accumulate_errors())
    return recs, I0, K, point, ev.bins


def stat_from(recs, idx, bins, is_dece):
    """Toolbox LaECE (class-mean of bin-weighted |prec_iou - mean score|) or
    D-ECE (class-agnostic |precision - mean score|) over image subset idx."""
    K = len(recs)
    nb = len(bins) - 1
    if is_dece:
        S, T, F = [], [], []
        for k in range(K):
            for i in idx:
                r = recs[k][i]
                if r is None:
                    continue
                S.append(r[0]); T.append(r[1]); F.append(r[2])
        if not S:
            return np.nan
        S = np.concatenate(S)
        T = np.concatenate(T).astype(bool)
        F = np.concatenate(F).astype(bool)
        tot = T.sum() + F.sum()
        if tot == 0:
            return np.nan
        ece = 0.0
        for b in range(nb):
            m = (bins[b] <= S) & (S <= bins[b + 1]) if b == 0 else (bins[b] < S) & (S <= bins[b + 1])
            ntp, nfp = (T & m).sum(), (F & m).sum()
            nd = ntp + nfp
            if nd == 0:
                continue
            ece += (nd / tot) * abs(ntp / nd - S[m].mean())
        return float(ece)
    vals = []
    for k in range(K):
        S, T, F, U, npig = [], [], [], [], 0
        for i in idx:
            r = recs[k][i]
            if r is None:
                continue
            S.append(r[0]); T.append(r[1]); F.append(r[2]); U.append(r[3]); npig += r[4]
        if not S or npig == 0:
            vals.append(np.nan)
            continue
        S = np.concatenate(S)
        T = np.concatenate(T).astype(bool)
        F = np.concatenate(F).astype(bool)
        U = np.concatenate(U)
        tot = T.sum() + F.sum()
        if tot == 0:
            vals.append(np.nan)
            continue
        bsum = 0.0
        for b in range(nb):
            m = (bins[b] <= S) & (S <= bins[b + 1]) if b == 0 else (bins[b] < S) & (S <= bins[b + 1])
            tpm = T & m
            nd = tpm.sum() + (F & m).sum()
            if nd == 0:
                continue
            bsum += (nd / tot) * abs(U[tpm].sum() / nd - S[m].mean())
        vals.append(np.nan if bsum == 0 else bsum)
    return float(np.nanmean(vals)) if not all(np.isnan(v) for v in vals) else np.nan


def run_cell(cell, B):
    art = json.loads((CAL / f"{cell}.json").read_text(encoding="utf-8"))
    inp = dict(art["inputs"])
    from detection_calibration.DetectionCalibration import DetectionCalibration
    # same class-space guard as run_calibration.py (TOOLBOX-PATCHES.md 2026-09-03)
    calval_cats = {c["id"] for c in json.loads(Path(inp["calval-gt"]).read_text(encoding="utf-8"))["categories"]}
    tmp = HERE / "tmp_calib_inputs"; tmp.mkdir(exist_ok=True)
    for key in ("calval-dets", "test-dets"):
        dets = json.loads(Path(inp[key]).read_text(encoding="utf-8"))
        keep = [d for d in dets if d["category_id"] in calval_cats]
        if len(keep) != len(dets):
            fp = tmp / f"ci__{cell}__{key}.json"; fp.write_text(json.dumps(keep), encoding="utf-8"); inp[key] = str(fp)
    rng = np.random.default_rng(SEED)
    boot_idx = None
    out = {"cell": cell, "B": B, "seed": SEED, "unit": "test image", "results": {}}
    t0 = time.time()
    for cal in CALIBRATORS:
        cm = DetectionCalibration(inp["calval-gt"], inp["test-gt"])
        calibrator, thresholds = _q(cm.fit, inp["calval-dets"], calibrator_type=cal)
        dets = _q(cm.transform, inp["test-dets"], calibrator, thresholds)
        res = {}
        for metric, (nb, tau, dece) in {"LaECE_0": (25, 0.0, False), "D_ECE": (10, 0.5, True)}.items():
            recs, I0, K, point, bins = per_image_records(inp["test-gt"], dets, nb, tau, dece)
            ref = art["results"][cal][metric]
            full = stat_from(recs, np.arange(I0), bins, dece)
            assert abs(point - ref) < 1e-6, f"{cell} {cal} {metric}: toolbox {point} vs artefact {ref}"
            assert abs(full - point) < 1e-6, f"{cell} {cal} {metric}: resample-stat {full} vs toolbox {point}"
            if boot_idx is None:
                boot_idx = rng.integers(0, I0, size=(B, I0))
            samples = np.array([stat_from(recs, boot_idx[b], bins, dece) for b in range(B)])
            res[metric] = {"point": point, "boot_mean": float(np.nanmean(samples)),
                           "ci95": [float(np.nanpercentile(samples, 2.5)),
                                    float(np.nanpercentile(samples, 97.5))],
                           "samples": samples.round(6).tolist()}
        out["results"][cal] = res
        print(f"[{cell}] {cal}: LaECE_0 {res['LaECE_0']['point']*100:.1f} "
              f"[{res['LaECE_0']['ci95'][0]*100:.1f}, {res['LaECE_0']['ci95'][1]*100:.1f}] | "
              f"D-ECE {res['D_ECE']['point']*100:.1f} "
              f"[{res['D_ECE']['ci95'][0]*100:.1f}, {res['D_ECE']['ci95'][1]*100:.1f}]", flush=True)
    out["paired_vs_identity"] = {}
    for cal in CALIBRATORS[1:]:
        d = {}
        for metric in ("LaECE_0", "D_ECE"):
            a = np.array(out["results"][cal][metric]["samples"])
            b0 = np.array(out["results"]["identity"][metric]["samples"])
            diff = a - b0
            d[metric] = {"point_diff": out["results"][cal][metric]["point"] - out["results"]["identity"][metric]["point"],
                         "ci95": [float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))],
                         "p_two_sided_boot": float(2 * min(np.nanmean(diff <= 0), np.nanmean(diff >= 0)))}
        out["paired_vs_identity"][cal] = d
    for cal in CALIBRATORS:
        for metric in ("LaECE_0", "D_ECE"):
            del out["results"][cal][metric]["samples"]
    out["wall_s"] = round(time.time() - t0, 1)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{cell}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("artefact ->", OUT / f"{cell}.json", f"({out['wall_s']} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    cells = [a.cell] if a.cell else sorted(p.stem for p in CAL.glob("*.json") if not (OUT / p.name).exists())
    for c in cells:
        run_cell(c, a.B)


if __name__ == "__main__":
    main()
