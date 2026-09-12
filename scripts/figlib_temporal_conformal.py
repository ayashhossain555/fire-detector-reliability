"""FIgLib pilot (CPU only, no inference): temporal evidence accumulation with a
conformal time-to-detection guarantee for tower-camera smoke alarms.

Inputs (all pre-existing, read-only):
  coco_gt/figlib_all.json          every frame: sequence, camera, offset_s
  figlib_scores/<run>.json         per-frame max score written by figlib_eval.py
  figlib_ttd.json                  the 4 camera folds (reused verbatim) and the
                                   per-frame baselines this pilot is compared to
  operating_points.json            shipped source F1-optimal threshold t_src

Per sequence the frames are ordered by offset_s; post-plume frame j = 0, 1, ...
are the frames with offset_s >= 0 (j = 0 is the first visible plume). Evidence
e_t = max detection score in frame t (smoke class for D-Fire models, any class
for Pyro-SDIS models, exactly as figlib_eval.py). Alarm statistics s_t:
  perframe      s_t = e_t                                          (R0 / R1)
  ema_h         s_t = l*s_{t-1} + (1-l)*e_t, l = 2^(-1/h), s_0 = e_0     (R2)
  k_of_n        s_t = k-th largest of e over the last n frames (zero-padded
                before the sequence start); s_t >= tau  <=>  at least k of
                the last n frames exceed tau                            (R2)
Every rule is "alarm when s_t >= tau", so thresholds are set the same way for
all of them:
  R0  tau = t_src (per-frame only; pooled over all 520 sequences).
  R1/R2 matched FAR: tau = (1-FAR) quantile of s_t over PRE-plume frames of the
      calibration cameras; evaluated on the held-out cameras (the 4 camera
      folds of figlib_ttd.json; fold mean and pooled held-out, each sequence
      held out exactly once).
  R3 conformal TTD: nonconformity M_D = max_{j<=D} s_j over the first D+1
      post-plume frames; tau = k-th smallest M_D on the calibration
      sequences, k = floor((n+1)*alpha) (split conformal), so that for
      exchangeable test sequences P(detected within D frames) >= 1-alpha.
      Realised miss-within-D on held-out CAMERAS (camera-disjoint) vs a
      random sequence split of the same size (exchangeable control, seed 0,
      10 repeats).
TTD = offset_s/60 of the first post-plume alarm frame (minutes, as in
figlib_ttd.json); "missed" = no post-plume alarm at all; frame FAR = fraction
of pre-plume frames alarmed; sequence FA = any pre-plume alarm. Delta is
counted in post-plume frame index (nominal 1 frame = 1 min; 71.6 % of
consecutive gaps are exactly 60 s).
Writes figlib_temporal_conformal.json and figlib_temporal_conformal.md.
Credit: HPWREN (https://www.hpwren.ucsd.edu/).
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
GT = HERE / "coco_gt" / "figlib_all.json"
SCORES = HERE / "figlib_scores"
TTD_JSON = HERE / "figlib_ttd.json"
OPS = HERE / "operating_points.json"
OUT_JSON = HERE / "figlib_temporal_conformal.json"
OUT_MD = HERE / "figlib_temporal_conformal.md"
RUNS = ["v8s_dfire_s3407", "y11s_dfire_s3407", "rtdetrl_dfire_s3407",
        "v8s_pyrosdis_s3407", "y11s_pyrosdis_s3407", "rtdetrl_pyrosdis_s3407", "rtdetrl_pyrosdis_s3407_p60"]
FARS = [0.01, 0.05, 0.10]
SEQ_FAS = [0.05, 0.10, 0.20, 0.30]  # matched SEQUENCE-level pre-plume false-alarm fraction
ALPHAS = [0.05, 0.10, 0.20]
DELTAS = [5, 10, 20]
HALF_LIVES = [1, 2, 4, 8]
KOFN = [(2, 3), (3, 3), (2, 5), (3, 5)]
SEED_RANDOM = 0
N_RANDOM = 10
STATS = ["perframe"] + [f"ema_h{h}" for h in HALF_LIVES] + [f"k{k}_of_n{n}" for k, n in KOFN]
MET = ("false_alarm_rate", "missed_frac", "ttd_min_median", "ttd_min_p90", "detected_within_10min_frac", "sequence_false_alarm_frac")


def load_sequences():
    g = json.loads(GT.read_text(encoding="utf-8"))
    by = defaultdict(list)
    for f in g["images"]:
        by[f["sequence"]].append(f)
    seqs = []
    for name in sorted(by):
        fr = sorted(by[name], key=lambda x: x["offset_s"])
        off = np.array([f["offset_s"] for f in fr])
        seqs.append({"name": name, "camera": fr[0]["camera"], "ids": [str(f["id"]) for f in fr],
                     "offsets": off, "p": int(np.searchsorted(off, 0))})  # p = index of the first post-plume frame
    return seqs


def statistics(e):
    out = {"perframe": e}
    for h in HALF_LIVES:
        lam = 0.5 ** (1.0 / h)
        s = np.empty_like(e)
        s[0] = e[0]
        for t in range(1, len(e)):
            s[t] = lam * s[t - 1] + (1 - lam) * e[t]
        out[f"ema_h{h}"] = s
    for k, n in KOFN:
        pad = np.concatenate([np.zeros(n - 1), e])
        win = np.lib.stride_tricks.sliding_window_view(pad, n)
        out[f"k{k}_of_n{n}"] = np.sort(win, axis=1)[:, -k]
    return out


def build(run, seqs):
    raw = json.loads((SCORES / f"{run}.json").read_text(encoding="utf-8"))
    key = "max_smoke" if "_dfire_" in run else "max"
    feats = {st: [] for st in STATS}
    for sq in seqs:
        e = np.array([raw[i][key] for i in sq["ids"]], dtype=float)
        for st, s in statistics(e).items():
            post = s[sq["p"]:]
            feats[st].append({"pre": s[:sq["p"]], "post": post, "off": sq["offsets"][sq["p"]:],
                              "M": {D: float(post[:D + 1].max()) for D in DELTAS}})
    return key, feats


def first_alarm(Fi, tau):
    al = np.flatnonzero(Fi["post"] >= tau)
    return (float(Fi["off"][al[0]] / 60.0), int(al[0])) if len(al) else None


def summarise(F, idx, taus):
    """Sequence-level metrics on sequence indices idx, sequence i thresholded at taus[i]."""
    pre = [F[i]["pre"] >= taus[i] for i in idx if len(F[i]["pre"])]
    pre = np.concatenate(pre) if pre else np.array([])
    first = [first_alarm(F[i], taus[i]) for i in idx]
    det = [x for x in first if x is not None]
    n = len(idx)
    r = {"n_sequences": n, "n_pre_frames": int(len(pre)),
         "false_alarm_rate": float(pre.mean()) if len(pre) else None,
         "sequence_false_alarm_frac": sum(1 for i in idx if len(F[i]["pre"]) and (F[i]["pre"] >= taus[i]).any()) / n,
         "missed_frac": sum(1 for x in first if x is None) / n,
         "ttd_min_median": float(np.median([d[0] for d in det])) if det else None,
         "ttd_min_p90": float(np.percentile([d[0] for d in det], 90)) if det else None,
         "detected_within_10min_frac": sum(1 for d in det if d[0] <= 10) / n}
    for D in DELTAS:
        r[f"miss_within_{D}f"] = sum(1 for x in first if x is None or x[1] > D) / n
    return r


def evaluate(F, idx, tau):
    return summarise(F, idx, {i: tau for i in idx})


def conformal_tau(F, idx, D, alpha):
    m = np.sort([F[i]["M"][D] for i in idx])
    k = max(int(np.floor((len(m) + 1) * alpha)), 1)
    return float(m[k - 1])


def far_tau(F, idx, far):
    return float(np.quantile(np.concatenate([F[i]["pre"] for i in idx if len(F[i]["pre"])]), 1 - far))


def cp_ci(x, n, level=0.95):
    lo = beta.ppf((1 - level) / 2, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(1 - (1 - level) / 2, x + 1, n - x) if x < n else 1.0
    return [float(lo), float(hi)]


def fold_agg(rows, key):
    v = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
    if not len(v):
        return None
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return {"mean": float(v.mean()), "sd": sd, "min": float(v.min()), "max": float(v.max()),
            "ci95_over_folds": [float(v.mean() - 1.96 * sd / np.sqrt(len(v))), float(v.mean() + 1.96 * sd / np.sqrt(len(v)))]}


def main():
    t0 = time.time()
    seqs = load_sequences()
    ttd = json.loads(TTD_JSON.read_text(encoding="utf-8"))
    folds = [set(f["cameras"]) for f in ttd["folds"]]
    fold_of = [next(fi for fi, c in enumerate(folds) if s["camera"] in c) for s in seqs]
    ops = json.loads(OPS.read_text(encoding="utf-8"))["cells"]
    all_idx = list(range(len(seqs)))
    n_seq = len(seqs)
    rng = np.random.default_rng(SEED_RANDOM)
    rand_splits = []
    for _ in range(N_RANDOM):
        perm = rng.permutation(n_seq)
        rand_splits.append((perm[:n_seq // 4].tolist(), perm[n_seq // 4:].tolist()))
    out = {"protocol": __doc__, "n_sequences": n_seq, "n_frames": sum(len(s["ids"]) for s in seqs),
           "n_cameras": len({s["camera"] for s in seqs}), "folds": ttd["folds"],
           "sequences_per_fold": [sum(1 for f in fold_of if f == fi) for fi in range(4)],
           "statistics": STATS, "fars": FARS, "alphas": ALPHAS, "deltas_frames": DELTAS,
           "random_control": {"seed": SEED_RANDOM, "n_repeats": N_RANDOM, "held_out_size": n_seq // 4},
           "runs": {}, "summary": {}}
    lines = []
    for run in RUNS:
        if not (SCORES / f"{run}.json").exists():
            print(f"[skip] {run}: no score file")
            continue
        key, feats = build(run, seqs)
        t_src = next((o["source"]["t_f1opt"] for c, o in ops.items() if c.startswith(run + "__to__")), None)
        R = {"score_used": key, "t_src": t_src,
             "R0_t_src": evaluate(feats["perframe"], all_idx, t_src) if t_src is not None else None,
             "matched_far": {}, "conformal_ttd": {}}
        # ---- R1 / R2 at matched FAR, camera-disjoint
        for st in STATS:
            F = feats[st]
            R["matched_far"][st] = {}
            for far in FARS:
                rows, taus = [], {}
                for fi in range(4):
                    te = [i for i in all_idx if fold_of[i] == fi]
                    ca = [i for i in all_idx if fold_of[i] != fi]
                    tau = far_tau(F, ca, far)
                    rows.append({"fold": fi, "tau": tau, **evaluate(F, te, tau)})
                    taus.update({i: tau for i in te})
                pooled = summarise(F, all_idx, taus)
                R["matched_far"][st][str(far)] = {
                    "fold_mean": {k: (fold_agg(rows, k) or {}).get("mean") for k in MET + tuple(f"miss_within_{D}f" for D in DELTAS)},
                    "pooled_heldout": pooled, "folds": rows}
        # ---- R1 / R2 at matched SEQUENCE-level false alarm (threshold = (1-FA) quantile of the
        #      per-sequence max pre-plume statistic on calibration cameras; sequences with no
        #      pre-plume frames excluded from the fit), camera-disjoint, pooled held-out
        R["matched_seq_fa"] = {}
        for st in STATS:
            F = feats[st]
            R["matched_seq_fa"][st] = {}
            for fa in SEQ_FAS:
                taus, rows = {}, []
                for fi in range(4):
                    te = [i for i in all_idx if fold_of[i] == fi]
                    ca = [i for i in all_idx if fold_of[i] != fi and len(F[i]["pre"])]
                    tau = float(np.quantile([F[i]["pre"].max() for i in ca], 1 - fa))
                    rows.append({"fold": fi, "tau": tau, **evaluate(F, te, tau)})
                    taus.update({i: tau for i in te})
                R["matched_seq_fa"][st][str(fa)] = {"pooled_heldout": summarise(F, all_idx, taus), "folds": rows}
        # ---- R3 conformal time-to-detection guarantee
        for st in STATS:
            F = feats[st]
            R["conformal_ttd"][st] = {}
            for alpha in ALPHAS:
                for D in DELTAS:
                    mk = f"miss_within_{D}f"
                    rows, pooled_miss = [], 0
                    for fi in range(4):
                        te = [i for i in all_idx if fold_of[i] == fi]
                        ca = [i for i in all_idx if fold_of[i] != fi]
                        tau = conformal_tau(F, ca, D, alpha)
                        rows.append({"fold": fi, "tau": tau, **evaluate(F, te, tau)})
                        pooled_miss += sum(1 for i in te if F[i]["M"][D] < tau)
                    rnd = []
                    for te, ca in rand_splits:
                        tau = conformal_tau(F, ca, D, alpha)
                        rnd.append(sum(1 for i in te if F[i]["M"][D] < tau) / len(te))
                    R["conformal_ttd"][st][f"alpha={alpha},delta={D}"] = {
                        "alpha": alpha, "delta_frames": D, "target_miss_within_delta": alpha,
                        "camera_disjoint": {
                            "miss_within_delta": fold_agg(rows, mk),
                            "pooled_heldout_miss_within_delta": pooled_miss / n_seq,
                            "pooled_heldout_ci95_clopper_pearson": cp_ci(pooled_miss, n_seq),
                            "guarantee_holds_pooled": pooled_miss / n_seq <= alpha,
                            "violation_significant_cp95": cp_ci(pooled_miss, n_seq)[0] > alpha,
                            "guarantee_holds_every_fold": all(r[mk] <= alpha for r in rows),
                            "n_folds_violating": sum(1 for r in rows if r[mk] > alpha),
                            "false_alarm_rate": fold_agg(rows, "false_alarm_rate"),
                            "sequence_false_alarm_frac": fold_agg(rows, "sequence_false_alarm_frac"),
                            "missed_frac": fold_agg(rows, "missed_frac"),
                            "ttd_min_median": fold_agg(rows, "ttd_min_median"),
                            "ttd_min_p90": fold_agg(rows, "ttd_min_p90"),
                            "tau": fold_agg(rows, "tau"), "folds": rows},
                        "random_split": {"miss_within_delta_mean": float(np.mean(rnd)), "miss_within_delta_max": float(np.max(rnd)),
                                         "guarantee_holds_mean": bool(float(np.mean(rnd)) <= alpha)}}
        out["runs"][run] = R
        # ---- summary block
        S = {"temporal_gain_at_matched_far": {}, "conformal": {}}
        for far in FARS:
            pf = R["matched_far"]["perframe"][str(far)]["pooled_heldout"]
            best = min((st for st in STATS if st != "perframe"),
                       key=lambda st: R["matched_far"][st][str(far)]["pooled_heldout"]["missed_frac"])
            bt = R["matched_far"][best][str(far)]["pooled_heldout"]
            S["temporal_gain_at_matched_far"][str(far)] = {
                "perframe": {k: pf[k] for k in MET},
                "best_temporal_by_missed_frac": best, "best_temporal_metrics": {k: bt[k] for k in MET},
                "delta_missed_frac": bt["missed_frac"] - pf["missed_frac"],
                "delta_ttd_min_median": (bt["ttd_min_median"] - pf["ttd_min_median"]) if (bt["ttd_min_median"] is not None and pf["ttd_min_median"] is not None) else None,
                "delta_detected_within_10min_frac": bt["detected_within_10min_frac"] - pf["detected_within_10min_frac"],
                "all_statistics": {st: {k: R["matched_far"][st][str(far)]["pooled_heldout"][k] for k in MET + ("miss_within_5f", "miss_within_10f", "miss_within_20f")} for st in STATS}}
        S["temporal_gain_at_matched_seq_fa"] = {}
        for fa in SEQ_FAS:
            pf = R["matched_seq_fa"]["perframe"][str(fa)]["pooled_heldout"]
            best = min((st for st in STATS if st != "perframe"),
                       key=lambda st: R["matched_seq_fa"][st][str(fa)]["pooled_heldout"]["missed_frac"])
            bt = R["matched_seq_fa"][best][str(fa)]["pooled_heldout"]
            S["temporal_gain_at_matched_seq_fa"][str(fa)] = {
                "perframe": {k: pf[k] for k in MET}, "best_temporal_by_missed_frac": best, "best_temporal_metrics": {k: bt[k] for k in MET},
                "delta_missed_frac": bt["missed_frac"] - pf["missed_frac"],
                "delta_ttd_min_median": (bt["ttd_min_median"] - pf["ttd_min_median"]) if (bt["ttd_min_median"] is not None and pf["ttd_min_median"] is not None) else None,
                "delta_detected_within_10min_frac": bt["detected_within_10min_frac"] - pf["detected_within_10min_frac"],
                "all_statistics": {st: {k: R["matched_seq_fa"][st][str(fa)]["pooled_heldout"][k] for k in MET} for st in STATS}}
        for alpha in ALPHAS:
            for D in DELTAS:
                kk = f"alpha={alpha},delta={D}"
                cd = lambda st: R["conformal_ttd"][st][kk]["camera_disjoint"]
                pf = cd("perframe")
                cands = [st for st in STATS if st != "perframe" and cd(st)["guarantee_holds_pooled"]]
                best = min(cands, key=lambda st: cd(st)["false_alarm_rate"]["mean"]) if cands else None
                S["conformal"][kk] = {
                    "perframe": {"pooled_miss_within_delta": pf["pooled_heldout_miss_within_delta"], "ci95": pf["pooled_heldout_ci95_clopper_pearson"],
                                 "holds_pooled": pf["guarantee_holds_pooled"], "n_folds_violating": pf["n_folds_violating"],
                                 "false_alarm_rate_mean": pf["false_alarm_rate"]["mean"], "sequence_false_alarm_mean": pf["sequence_false_alarm_frac"]["mean"],
                                 "random_split_miss": R["conformal_ttd"]["perframe"][kk]["random_split"]["miss_within_delta_mean"]},
                    "best_temporal_by_far_with_guarantee": best,
                    "best_temporal": None if best is None else {
                        "pooled_miss_within_delta": cd(best)["pooled_heldout_miss_within_delta"], "ci95": cd(best)["pooled_heldout_ci95_clopper_pearson"],
                        "n_folds_violating": cd(best)["n_folds_violating"], "false_alarm_rate_mean": cd(best)["false_alarm_rate"]["mean"],
                        "sequence_false_alarm_mean": cd(best)["sequence_false_alarm_frac"]["mean"]},
                    "n_statistics_guarantee_holds_pooled": sum(1 for st in STATS if cd(st)["guarantee_holds_pooled"]),
                    "n_statistics_violation_significant_cp95": sum(1 for st in STATS if cd(st)["violation_significant_cp95"]),
                    "all_statistics": {st: {"pooled_miss": cd(st)["pooled_heldout_miss_within_delta"], "n_folds_violating": cd(st)["n_folds_violating"],
                                            "violation_significant_cp95": cd(st)["violation_significant_cp95"],
                                            "far_mean": cd(st)["false_alarm_rate"]["mean"], "seq_fa_mean": cd(st)["sequence_false_alarm_frac"]["mean"],
                                            "random_miss": R["conformal_ttd"][st][kk]["random_split"]["miss_within_delta_mean"]} for st in STATS}}
        out["summary"][run] = S
        # ---- compact print
        lines.append(f"\n### {run}  (score={key}, t_src={t_src})")
        r0 = R["R0_t_src"]
        if r0:
            lines.append(f"R0 at t_src (all 520 seq): FAR {r0['false_alarm_rate']:.3f} missed {r0['missed_frac']:.3f} TTD med {r0['ttd_min_median']:.1f} p90 {r0['ttd_min_p90']:.1f} seqFA {r0['sequence_false_alarm_frac']:.3f}")
        lines.append("| stat | FAR | realised FAR | missed | TTD med | TTD p90 | <=10min | seqFA | miss<=5f | miss<=10f | miss<=20f |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for far in FARS:
            for st in STATS:
                p = R["matched_far"][st][str(far)]["pooled_heldout"]
                lines.append(f"| {st} | {far} | {p['false_alarm_rate']:.3f} | {p['missed_frac']:.3f} | {p['ttd_min_median']:.1f} | {p['ttd_min_p90']:.1f} | {p['detected_within_10min_frac']:.3f} | {p['sequence_false_alarm_frac']:.3f} | {p['miss_within_5f']:.3f} | {p['miss_within_10f']:.3f} | {p['miss_within_20f']:.3f} |")
        lines.append("| stat | seq-FA target | realised seqFA | frame FAR | missed | TTD med | TTD p90 | <=10min |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for fa in SEQ_FAS:
            for st in STATS:
                p = R["matched_seq_fa"][st][str(fa)]["pooled_heldout"]
                lines.append(f"| {st} | {fa} | {p['sequence_false_alarm_frac']:.3f} | {p['false_alarm_rate']:.3f} | {p['missed_frac']:.3f} | {p['ttd_min_median']:.1f} | {p['ttd_min_p90']:.1f} | {p['detected_within_10min_frac']:.3f} |")
        lines.append("| stat | alpha | delta | held-out miss (pooled) [CP95] | fold mean (min-max) | folds>alpha | FAR mean | seqFA mean | random miss |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for alpha in ALPHAS:
            for D in DELTAS:
                kk = f"alpha={alpha},delta={D}"
                for st in STATS:
                    c = R["conformal_ttd"][st][kk]["camera_disjoint"]
                    m = c["miss_within_delta"]
                    lines.append(f"| {st} | {alpha} | {D} | {c['pooled_heldout_miss_within_delta']:.3f} [{c['pooled_heldout_ci95_clopper_pearson'][0]:.3f},{c['pooled_heldout_ci95_clopper_pearson'][1]:.3f}] | {m['mean']:.3f} ({m['min']:.3f}-{m['max']:.3f}) | {c['n_folds_violating']} | {c['false_alarm_rate']['mean']:.3f} | {c['sequence_false_alarm_frac']['mean']:.3f} | {R['conformal_ttd'][st][kk]['random_split']['miss_within_delta_mean']:.3f} |")
        print(f"{run} done ({time.time() - t0:.1f}s)", flush=True)
    out["runtime_s"] = time.time() - t0
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# FIgLib temporal accumulation + conformal time-to-detection (CPU pilot)",
          f"Generated by `figlib_temporal_conformal.py`; artefact `figlib_temporal_conformal.json`; runtime {out['runtime_s']:.1f} s.",
          "Matched-FAR rows: threshold fitted on the calibration cameras' pre-plume frames, evaluated on the held-out cameras, pooled over the 4 folds (each sequence held out once).",
          "Conformal rows: tau = split-conformal quantile of M_delta on calibration cameras; the held-out miss-within-delta should be <= alpha if the guarantee transfers across cameras.",
          *lines]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(lines))
    print("artefact ->", OUT_JSON, "\nruntime", f"{out['runtime_s']:.1f}s")


if __name__ == "__main__":
    main()
