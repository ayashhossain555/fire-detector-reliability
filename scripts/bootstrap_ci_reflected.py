"""Reflected (basic) and BCa bootstrap intervals for the identity-map LaECE_0 and D_ECE.

Referee item: the percentile intervals in analysis/calibration_ci/<cell>.json are the
only interval reported so far; for a bounded, skewed statistic near its floor the
percentile interval can be biased. This script re-runs the SAME image-level bootstrap
(resampling logic copied verbatim from bootstrap_ci.py: same toolbox per-image records,
same statistic, same seed 3407, same draw of the B x I0 index matrix, B = 1000) for
the identity calibrator only, on

  * the 37 in-domain cells  - every run scored on its own source test split
    (d_fire_test AND d_fire_test_dedup for the 13 D-Fire-trained runs incl. dfiredd /
    dfiresub, pyro_sdis_caltest for the 11 Pyro-SDIS-trained runs incl. the _p60
    variant; the brief said 36 - no run is dropped), and
  * the 12 core cross-domain cells of the decomposition - the six seed-3407 core runs
    (v8s/y11s/rtdetrl x dfire/pyrosdis) x their two labelled cross targets, smoke-only
    for the Pyro-SDIS runs,

and reports per cell and per metric: point estimate, bootstrap mean, bootstrap SE,
percentile 95 % interval (asserted equal to calibration_ci/<cell>.json ci95 to 1e-9),
basic / reflected interval 2*theta_hat - [q97.5, q2.5], and the BCa interval
(bias-correction z0 from the fraction of replicates below the point; acceleration a
from a leave-one-image-out jackknife of the same statistic).

Nothing in bootstrap_ci.py or calibration_ci/ is touched: outputs go to
analysis/calibration_ci_reflected/<cell>.json, summary to
analysis/bootstrap_ci_reflected.json / .md; temp files use the prefix "cir__".

2026-09-14 addition (RT-DETR-x runs rtdetrx_dfire_s3407 / rtdetrx_pyrosdis_s3407, 26-run matrix):
their in-domain cells join the in-domain list by the existing rule (the count assert is now
derived from matrix_summary.json: 2 x D-Fire-trained runs + Pyro-SDIS-trained runs = 40), and
their four labelled cross cells are listed under the role "extra_cross" (EXTRA_CROSS_RUNS) so
that --cells can address them; the 12 core_cross cells and every existing output are unchanged.

Usage: python analysis/bootstrap_ci_reflected.py --cells A B    # verification pair
       python analysis/bootstrap_ci_reflected.py --all           # every listed cell lacking output
       python analysis/bootstrap_ci_reflected.py --summary       # (re)build the summary only
"""
import argparse
import contextlib
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CAL = HERE / "calibration"
REF = HERE / "calibration_ci"          # read-only reference (percentile ci95)
OUT = HERE / "calibration_ci_reflected"
SUMMARY_JSON = HERE / "bootstrap_ci_reflected.json"
SUMMARY_MD = HERE / "bootstrap_ci_reflected.md"
MATRIX = HERE / "matrix_summary.json"
SEED = 3407
B_DEFAULT = 1000
METRICS = {"LaECE_0": (25, 0.0, False), "D_ECE": (10, 0.5, True)}
CORE_RUNS = ["v8s_dfire_s3407", "y11s_dfire_s3407", "rtdetrl_dfire_s3407",
             "v8s_pyrosdis_s3407", "y11s_pyrosdis_s3407", "rtdetrl_pyrosdis_s3407"]
# 2026-09-14: RT-DETR-x runs; their two labelled cross targets each get role "extra_cross" (not core).
EXTRA_CROSS_RUNS = ["rtdetrx_dfire_s3407", "rtdetrx_pyrosdis_s3407"]


def _q(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ---------------------------------------------------------------------------------------------
# per_image_records() and stat_from() are copied VERBATIM from bootstrap_ci.py (2026-09-03,
# unchanged since) so the resampled statistic is the same toolbox statistic.
# ---------------------------------------------------------------------------------------------
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
# ------------------------------------------------------------------- end of verbatim copy


def cell_list():
    """36 in-domain cells + 12 core cross cells, with the matrix_summary 'kind' label."""
    m = json.loads(MATRIX.read_text(encoding="utf-8"))
    cells = []
    for run in sorted(m["runs"]):
        if "_dfire" in run:
            cells += [(f"{run}__to__d_fire_test", "in_domain"), (f"{run}__to__d_fire_test_dedup", "in_domain")]
        elif "_pyrosdis" in run:
            cells += [(f"{run}__to__pyro_sdis_caltest", "in_domain")]
    for run in CORE_RUNS:
        if "_dfire" in run:
            cells += [(f"{run}__to__pyro_sdis_caltest", "core_cross"), (f"{run}__to__thesis_test", "core_cross")]
        else:
            cells += [(f"{run}__to__d_fire_test_smokeonly", "core_cross"), (f"{run}__to__thesis_test_smokeonly", "core_cross")]
    for run in EXTRA_CROSS_RUNS:
        if run not in m["runs"]:
            continue
        if "_dfire" in run:
            cells += [(f"{run}__to__pyro_sdis_caltest", "extra_cross"), (f"{run}__to__thesis_test", "extra_cross")]
        else:
            cells += [(f"{run}__to__d_fire_test_smokeonly", "extra_cross"), (f"{run}__to__thesis_test_smokeonly", "extra_cross")]
    out = []
    for c, role in cells:
        assert (CAL / f"{c}.json").exists(), f"missing calibration cell {c}"
        out.append({"cell": c, "role": role, "matrix_summary_kind": m["cells"][c]["kind"]})
    # 13 D-Fire-trained runs x 2 splits + 11 Pyro-SDIS-trained runs (incl. rtdetrl_pyrosdis_s3407_p60) = 37,
    # one more than the "36" of the brief: every run of matrix_summary.json is kept, none is dropped.
    # 2026-09-14: 14 D-Fire-trained x 2 + 12 Pyro-SDIS-trained = 40 with the two RT-DETR-x runs; derived, not hard-coded.
    n_df = sum(1 for r in m["runs"] if "_dfire" in r); n_py = sum(1 for r in m["runs"] if "_pyrosdis" in r)
    assert len([c for c in out if c["role"] == "in_domain"]) == 2 * n_df + n_py, f"expected {2 * n_df + n_py} in-domain cells"
    assert len([c for c in out if c["role"] == "core_cross"]) == 12, "expected 12 core cross cells"
    return out


def intervals(samples, point, jack, alpha=0.05):
    """Percentile, basic (reflected) and BCa intervals from bootstrap replicates."""
    s = np.asarray(samples, dtype=float)
    ok = ~np.isnan(s)
    n_nan = int((~ok).sum())
    lo_p, hi_p = float(np.nanpercentile(s, 100 * alpha / 2)), float(np.nanpercentile(s, 100 * (1 - alpha / 2)))
    basic = [2 * point - hi_p, 2 * point - lo_p]
    se = float(np.nanstd(s, ddof=1))
    # BCa
    frac_below = float(np.mean(s[ok] < point))
    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    bca = {"z0": None, "a": None, "ci95": None, "alpha_lo": None, "alpha_hi": None, "note": None}
    if 0.0 < frac_below < 1.0:
        z0 = float(norm.ppf(frac_below))
        if jack is not None:
            j = np.asarray(jack, dtype=float)
            j = j[~np.isnan(j)]
            d = j.mean() - j
            den = 6.0 * (np.sum(d ** 2) ** 1.5)
            a = float(np.sum(d ** 3) / den) if den > 0 else 0.0
            a_note = f"jackknife over {len(j)} images"
        else:
            a, a_note = 0.0, "a=0 (no jackknife)"
        al = float(norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo))))
        ah = float(norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi))))
        bca.update({"z0": z0, "a": a, "alpha_lo": al, "alpha_hi": ah,
                    "ci95": [float(np.nanpercentile(s, 100 * al)), float(np.nanpercentile(s, 100 * ah))],
                    "note": a_note})
    else:
        bca["note"] = f"undefined: fraction of replicates below the point is {frac_below} (z0 infinite)"
    return {"point": point, "boot_mean": float(np.nanmean(s)), "boot_se": se, "n_nan_replicates": n_nan,
            "fraction_replicates_below_point": frac_below,
            "percentile_ci95": [lo_p, hi_p], "basic_ci95": basic, "bca": bca}


def run_cell(entry, B, do_jackknife=True):
    cell = entry["cell"]
    art = json.loads((CAL / f"{cell}.json").read_text(encoding="utf-8"))
    inp = dict(art["inputs"])
    from detection_calibration.DetectionCalibration import DetectionCalibration
    # same class-space guard as run_calibration.py / bootstrap_ci.py (own temp prefix)
    calval_cats = {c["id"] for c in json.loads(Path(inp["calval-gt"]).read_text(encoding="utf-8"))["categories"]}
    tmp = HERE / "tmp_calib_inputs"; tmp.mkdir(exist_ok=True)
    for key in ("calval-dets", "test-dets"):
        dets = json.loads(Path(inp[key]).read_text(encoding="utf-8"))
        keep = [d for d in dets if d["category_id"] in calval_cats]
        if len(keep) != len(dets):
            fp = tmp / f"cir__{cell}__{key}.json"; fp.write_text(json.dumps(keep), encoding="utf-8"); inp[key] = str(fp)
    ref = None
    if (REF / f"{cell}.json").exists():
        ref = json.loads((REF / f"{cell}.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    boot_idx = None
    t0 = time.time()
    cm = DetectionCalibration(inp["calval-gt"], inp["test-gt"])
    calibrator, thresholds = _q(cm.fit, inp["calval-dets"], calibrator_type="identity")
    dets = _q(cm.transform, inp["test-dets"], calibrator, thresholds)
    out = {"cell": cell, "role": entry["role"], "matrix_summary_kind": entry["matrix_summary_kind"],
           "calibrator": "identity", "B": B, "seed": SEED, "unit": "test image",
           "resampling": "copied verbatim from bootstrap_ci.py (same rng draw, same statistic)",
           "created": datetime.now().isoformat(timespec="seconds"), "results": {}}
    for metric, (nb, tau, dece) in METRICS.items():
        tm = time.time()
        recs, I0, K, point, bins = per_image_records(inp["test-gt"], dets, nb, tau, dece)
        ref_point = art["results"]["identity"][metric]
        full = stat_from(recs, np.arange(I0), bins, dece)
        assert abs(point - ref_point) < 1e-6, f"{cell} {metric}: toolbox {point} vs artefact {ref_point}"
        assert abs(full - point) < 1e-6, f"{cell} {metric}: resample-stat {full} vs toolbox {point}"
        if boot_idx is None:
            boot_idx = rng.integers(0, I0, size=(B, I0))
        samples = np.array([stat_from(recs, boot_idx[b], bins, dece) for b in range(B)])
        t_boot = time.time() - tm
        jack = None
        t_jack = None
        if do_jackknife:
            tj = time.time()
            all_idx = np.arange(I0)
            jack = np.array([stat_from(recs, np.delete(all_idx, i), bins, dece) for i in range(I0)])
            t_jack = time.time() - tj
        res = intervals(samples, point, jack)
        res.update({"n_images": int(I0), "n_classes": int(K), "seconds_bootstrap": round(t_boot, 1),
                    "seconds_jackknife": None if t_jack is None else round(t_jack, 1)})
        if jack is not None:
            res["jackknife"] = {"n": int(I0), "n_nan": int(np.isnan(jack).sum()),
                                "mean": float(np.nanmean(jack)), "se_jackknife": float(np.sqrt((I0 - 1) / I0 * np.nansum((jack - np.nanmean(jack)) ** 2)))}
        if ref is not None:
            rc = ref["results"]["identity"][metric]
            d_lo = abs(res["percentile_ci95"][0] - rc["ci95"][0]); d_hi = abs(res["percentile_ci95"][1] - rc["ci95"][1])
            res["reproduces_calibration_ci"] = {"ref_ci95": rc["ci95"], "ref_point": rc["point"], "ref_B": ref["B"],
                                                "max_abs_diff": max(d_lo, d_hi), "point_abs_diff": abs(point - rc["point"]),
                                                "within_1e-9": bool(max(d_lo, d_hi) < 1e-9 and ref["B"] == B)}
        out["results"][metric] = res
        bca = res["bca"]["ci95"]
        print(f"[{cell}] {metric}: point {point*100:.2f} se {res['boot_se']*100:.2f} "
              f"pct [{res['percentile_ci95'][0]*100:.2f},{res['percentile_ci95'][1]*100:.2f}] "
              f"basic [{res['basic_ci95'][0]*100:.2f},{res['basic_ci95'][1]*100:.2f}] "
              f"bca {'[%.2f,%.2f]' % (bca[0]*100, bca[1]*100) if bca else 'undef'} "
              f"z0 {res['bca']['z0']} a {res['bca']['a']} | boot {t_boot:.0f}s jack {t_jack if t_jack is None else round(t_jack)}s "
              f"| repro {res.get('reproduces_calibration_ci', {}).get('within_1e-9')}", flush=True)
    out["wall_s"] = round(time.time() - t0, 1)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{cell}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("artefact ->", OUT / f"{cell}.json", f"({out['wall_s']} s)", flush=True)
    return out


def build_summary(entries):
    rows = []
    for e in entries:
        fp = OUT / f"{e['cell']}.json"
        if not fp.exists():
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        row = {"cell": e["cell"], "role": e["role"], "matrix_summary_kind": e["matrix_summary_kind"], "wall_s": d["wall_s"]}
        for metric in METRICS:
            r = d["results"][metric]
            row[metric] = {k: r[k] for k in ("point", "boot_mean", "boot_se", "percentile_ci95", "basic_ci95", "n_images", "n_nan_replicates",
                                             "fraction_replicates_below_point")}
            row[metric]["bca_ci95"] = r["bca"]["ci95"]; row[metric]["bca_z0"] = r["bca"]["z0"]; row[metric]["bca_a"] = r["bca"]["a"]
            row[metric]["bca_note"] = r["bca"]["note"]
            row[metric]["reproduces_calibration_ci_within_1e-9"] = r.get("reproduces_calibration_ci", {}).get("within_1e-9")
            row[metric]["reproduces_max_abs_diff"] = r.get("reproduces_calibration_ci", {}).get("max_abs_diff")
        rows.append(row)

    def flags(ci, point):
        if ci is None:
            return {"crosses_zero": None, "excludes_point": None}
        return {"crosses_zero": bool(ci[0] <= 0.0), "excludes_point": bool(point < ci[0] or point > ci[1])}

    counts = {}
    for metric in METRICS:
        c = {"n_cells": len(rows)}
        for kind in ("percentile_ci95", "basic_ci95", "bca_ci95"):
            cz = ex = 0; undefined = 0; cz_cells = []; ex_cells = []
            for row in rows:
                f = flags(row[metric][kind], row[metric]["point"])
                if f["crosses_zero"] is None:
                    undefined += 1; continue
                if f["crosses_zero"]: cz += 1; cz_cells.append(row["cell"])
                if f["excludes_point"]: ex += 1; ex_cells.append(row["cell"])
            c[kind] = {"n_crosses_zero": cz, "cells_crossing_zero": cz_cells, "n_excludes_point": ex, "cells_excluding_point": ex_cells,
                       "n_undefined": undefined}
        # width and shift diagnostics
        w_p = [row[metric]["percentile_ci95"][1] - row[metric]["percentile_ci95"][0] for row in rows]
        w_b = [row[metric]["bca_ci95"][1] - row[metric]["bca_ci95"][0] for row in rows if row[metric]["bca_ci95"]]
        sh = [(row[metric]["bca_ci95"][0] - row[metric]["percentile_ci95"][0], row[metric]["bca_ci95"][1] - row[metric]["percentile_ci95"][1])
              for row in rows if row[metric]["bca_ci95"]]
        c["percentile_width_mean"] = float(np.mean(w_p)); c["bca_width_mean"] = float(np.mean(w_b)) if w_b else None
        c["bca_minus_percentile_lower_mean"] = float(np.mean([s[0] for s in sh])) if sh else None
        c["bca_minus_percentile_upper_mean"] = float(np.mean([s[1] for s in sh])) if sh else None
        c["bca_minus_percentile_lower_max_abs"] = float(np.max([abs(s[0]) for s in sh])) if sh else None
        c["bca_minus_percentile_upper_max_abs"] = float(np.max([abs(s[1]) for s in sh])) if sh else None
        c["n_reproduces_calibration_ci_within_1e-9"] = int(sum(1 for row in rows if row[metric]["reproduces_calibration_ci_within_1e-9"]))
        c["max_abs_diff_vs_calibration_ci"] = float(max(row[metric]["reproduces_max_abs_diff"] for row in rows if row[metric]["reproduces_max_abs_diff"] is not None))
        c["bias_boot_mean_minus_point_mean"] = float(np.mean([row[metric]["boot_mean"] - row[metric]["point"] for row in rows]))
        c["n_boot_mean_above_point"] = int(sum(1 for row in rows if row[metric]["boot_mean"] > row[metric]["point"]))
        c["abs_a_max"] = float(max(abs(row[metric]["bca_a"]) for row in rows if row[metric]["bca_a"] is not None))
        c["abs_z0_max"] = float(max(abs(row[metric]["bca_z0"]) for row in rows if row[metric]["bca_z0"] is not None))
        counts[metric] = c
    summary = {"created": datetime.now().isoformat(timespec="seconds"), "B": B_DEFAULT, "seed": SEED, "calibrator": "identity",
               "unit": "test image", "n_cells_done": len(rows), "n_cells_planned": len(entries),
               "definitions": {"percentile_ci95": "[q2.5, q97.5] of the B replicates (nanpercentile), as bootstrap_ci.py",
                               "basic_ci95": "reflected: [2*point - q97.5, 2*point - q2.5]",
                               "bca_ci95": "BCa: z0 = Phi^-1(fraction of replicates < point); a = sum(d^3)/(6 sum(d^2)^1.5), d = mean(jack) - jack_i, "
                                           "leave-one-image-out jackknife of the same statistic; endpoints at Phi(z0 + (z0+z)/(1 - a (z0+z))), z = +-1.96",
                               "crosses_zero": "lower endpoint <= 0", "excludes_point": "point outside the interval",
                               "role": "in_domain = run on its own source test split (dfiredd/dfiresub runs are labelled 'cross' in matrix_summary.json "
                                       "but are in-domain by this definition); core_cross = six seed-3407 core runs x two labelled cross targets"},
               "counts": counts, "cells": rows,
               "wall_s_total": float(sum(r["wall_s"] for r in rows))}
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # markdown
    L = [f"# Reflected / BCa bootstrap intervals - identity map (B={B_DEFAULT}, seed {SEED}, image-level)", "",
         f"Written by `bootstrap_ci_reflected.py` on {summary['created']}. Source: `bootstrap_ci_reflected.json`. "
         f"Resampling copied verbatim from `bootstrap_ci.py`; per-cell artefacts in `calibration_ci_reflected/<cell>.json`.", "",
         f"Cells done: {len(rows)} / {len(entries)} ({sum(1 for e in entries if e['role'] == 'in_domain')} in-domain + "
         f"{sum(1 for e in entries if e['role'] == 'core_cross')} core cross + {sum(1 for e in entries if e['role'] == 'extra_cross')} extra cross). "
         "Values in percentage points (x100).", ""]
    for metric in METRICS:
        c = counts[metric]
        L += [f"## {metric}", "",
              f"- percentile interval reproduces `calibration_ci/<cell>.json` ci95 to 1e-9 in {c['n_reproduces_calibration_ci_within_1e-9']} / {c['n_cells']} cells "
              f"(max abs diff {c['max_abs_diff_vs_calibration_ci']:.2e}) - key `counts.{metric}.n_reproduces_calibration_ci_within_1e-9`",
              f"- cells where the interval crosses zero: percentile {c['percentile_ci95']['n_crosses_zero']}, basic {c['basic_ci95']['n_crosses_zero']}, "
              f"BCa {c['bca_ci95']['n_crosses_zero']} (BCa undefined in {c['bca_ci95']['n_undefined']}) - keys `counts.{metric}.<kind>.n_crosses_zero`",
              f"- cells where the interval excludes the point: percentile {c['percentile_ci95']['n_excludes_point']}, basic {c['basic_ci95']['n_excludes_point']}, "
              f"BCa {c['bca_ci95']['n_excludes_point']} - keys `counts.{metric}.<kind>.n_excludes_point`",
              f"- mean (boot_mean - point) = {c['bias_boot_mean_minus_point_mean']*100:+.3f} pp; boot_mean above point in {c['n_boot_mean_above_point']} / {c['n_cells']} cells",
              f"- mean width: percentile {c['percentile_width_mean']*100:.3f} pp, BCa {c['bca_width_mean']*100:.3f} pp; mean BCa shift of lower / upper endpoint vs percentile: "
              f"{c['bca_minus_percentile_lower_mean']*100:+.3f} / {c['bca_minus_percentile_upper_mean']*100:+.3f} pp (max |shift| {c['bca_minus_percentile_lower_max_abs']*100:.3f} / {c['bca_minus_percentile_upper_max_abs']*100:.3f} pp)",
              f"- max |a| = {c['abs_a_max']:.4f}, max |z0| = {c['abs_z0_max']:.3f}", "",
              "| cell | role | n_img | point | boot mean | SE | percentile | basic | BCa | z0 | a | basic flags | BCa flags |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in rows:
            r = row[metric]
            def fmt(ci):
                return "undef" if ci is None else f"[{ci[0]*100:.2f}, {ci[1]*100:.2f}]"
            def fl(ci):
                f = flags(ci, r["point"])
                if f["crosses_zero"] is None: return "-"
                return ("cross0 " if f["crosses_zero"] else "") + ("exclPt" if f["excludes_point"] else "") or "ok"
            z0s = "-" if r["bca_z0"] is None else "%+.3f" % r["bca_z0"]
            a_s = "-" if r["bca_a"] is None else "%+.4f" % r["bca_a"]
            L.append(f"| {row['cell']} | {row['role']} | {r['n_images']} | {r['point']*100:.2f} | {r['boot_mean']*100:.2f} | {r['boot_se']*100:.2f} | "
                     f"{fmt(r['percentile_ci95'])} | {fmt(r['basic_ci95'])} | {fmt(r['bca_ci95'])} | {z0s} | {a_s} | {fl(r['basic_ci95'])} | {fl(r['bca_ci95'])} |")
        L.append("")
    L += [f"Total wall time of the per-cell runs: {summary['wall_s_total']:.0f} s.", ""]
    SUMMARY_MD.write_text("\n".join(L), encoding="utf-8")
    print("summary ->", SUMMARY_JSON, "and", SUMMARY_MD, f"({len(rows)} cells)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--no-jackknife", action="store_true")
    a = ap.parse_args()
    entries = cell_list()
    by_name = {e["cell"]: e for e in entries}
    if a.cells:
        todo = [by_name[c] for c in a.cells]
    elif a.all:
        todo = [e for e in entries if not (OUT / f"{e['cell']}.json").exists()]
    else:
        todo = []
    t0 = time.time()
    for e in todo:
        run_cell(e, a.B, do_jackknife=not a.no_jackknife)
    if todo:
        print(f"ran {len(todo)} cells in {time.time() - t0:.0f} s")
    if a.summary or a.all:
        build_summary(entries)


if __name__ == "__main__":
    main()
