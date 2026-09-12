"""EAAI-revision figures (referee comments, 2026-09-12): core-cell replacements for fig2/fig5 and the
label-free threshold-transfer method figure. Writes NEW files only; make_figures.py is untouched.

Same conventions as make_figures.py: static, print-first matplotlib; validated hub palette
(categorical blue #2a78d6 / orange #eb6834 / aqua #1baf7a for YOLOv8 / YOLO11 / RT-DETR, sequential
blue ramp for magnitude, ink tokens for text); PNG + vector PDF in figures_paper/; an index JSON
naming the artefact + key each panel reads (figures_index_eaai.json, with the key numbers shown).

  fig2_decomposition_core   2 x 5 grid of 2 x 3 mini-heatmaps (threshold origin x confidence map),
                            LaECE_0 x100 in every cell, ten core seed-3407 cross cells
  fig5_paired_posthoc_core  forest plot, paired (calibrator - identity) LaECE_0 with 95 % CI, the
                            eleven forward (source-fitted) cells of the six core seed-3407 models
  fig8_labelfree            (a) box F1 on the target test: shipped t_src / kept-rate rule / labelled
                            oracle, 44 cells ordered by W1(per-image max score);
                            (b) unlabelled-adaptation-set size curve (kept-rate rule);
                            (c) QQ plot of TRUE-detection scores, source vs target calval, logit axes,
                            with the shared logit shift fitted by the prior-shift mixture (pilot artefact).
  fig9_reliability          2 x 2 detection reliability diagrams (LaECE_0 bins, 25 equal-width, IoU-weighted
                            precision, toolbox tau = 0 matching) for YOLOv8-s and RT-DETR-l trained on D-Fire,
                            in-domain (D-Fire test) vs shifted (Pyro-SDIS), at the source LRP-optimal threshold,
                            with the source-fitted Platt map (refitted with the toolbox, verified against the
                            stored operating thresholds) and a count histogram per bin
Sources: threshold_decomposition.json, calibration_ci/<cell>.json, label_free_threshold_refine.json,
label_free_threshold_pilot.json, detections/*.bbox.json + coco_gt/*.json (panel c, recomputed here
with operating_points.match via label_free_threshold_pilot; fig9 recomputed with the pinned toolbox
CalibrationCOCO exactly as calib_metrics.evaluate does), calibration/<cell>.json (fig9 thresholds),
matrix_summary.json (fig9 LaECE_0 assertion).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # figures never need the GPU; fig9 imports the toolbox (torch) for the Platt refit
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (Type 42), not Type 3, for Elsevier
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import label_free_threshold_pilot as pilot  # noqa: E402  (read-only reuse: op.match, paths, EPS)

OUT = HERE / "figures_paper"
OUT.mkdir(exist_ok=True)
FAM = {"v8": ("YOLOv8", "#2a78d6"), "y11": ("YOLO11", "#eb6834"), "rtdetr": ("RT-DETR", "#1baf7a")}
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8985", "#d9d8d3"
TARGET_LABEL = {"d_fire_test": "D-Fire test", "d_fire_test_smokeonly": "D-Fire smoke", "pyro_sdis_caltest": "Pyro-SDIS",
                "thesis_test": "Thesis set", "thesis_test_smokeonly": "Thesis smoke"}
RUN_LABEL = {"v8s_dfire_s3407": "YOLOv8-s / D-Fire", "y11s_dfire_s3407": "YOLO11-s / D-Fire", "rtdetrl_dfire_s3407": "RT-DETR-l / D-Fire",
             "v8s_pyrosdis_s3407": "YOLOv8-s / Pyro-SDIS", "y11s_pyrosdis_s3407": "YOLO11-s / Pyro-SDIS"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9, "axes.titleweight": "normal",
                     "figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
index = {}


def fam_of(run):
    return "rtdetr" if run.startswith("rtdetr") else "y11" if run.startswith("y11") else "v8"


def save(fig, name, sources, key_values):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    index[name] = {"sources": sources, "key_values": key_values}
    print("[fig]", name)


# ----------------------------------------------------------------------------- fig2 core
CORE_DECOMP = [  # (row, col) -> cell; columns = source -> target pairing, rows = v8s / y11s; last column RT-DETR
    ["v8s_dfire_s3407__to__pyro_sdis_caltest", "v8s_pyrosdis_s3407__to__d_fire_test_smokeonly", "v8s_pyrosdis_s3407__to__thesis_test_smokeonly",
     "v8s_dfire_s3407__to__thesis_test", "rtdetrl_dfire_s3407__to__pyro_sdis_caltest"],
    ["y11s_dfire_s3407__to__pyro_sdis_caltest", "y11s_pyrosdis_s3407__to__d_fire_test_smokeonly", "y11s_pyrosdis_s3407__to__thesis_test_smokeonly",
     "y11s_dfire_s3407__to__thesis_test", "rtdetrl_dfire_s3407__to__thesis_test"],
]
DEC_ROWS = [("source", "thr=source"), ("target", "thr=target")]
DEC_COLS = [("none", "map=none"), ("source\nPlatt", "map=source:platt"), ("target\nPlatt", "map=target:platt")]


def fig2_core(decomp):
    cells = decomp["cells"]
    fig, axes = plt.subplots(2, 5, figsize=(7.5, 3.9), gridspec_kw={"wspace": 0.30, "hspace": 0.85})
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", SEQ)
    vmax = 45.0
    kv = {}
    for i, row in enumerate(CORE_DECOMP):
        for j, cell in enumerate(row):
            ax = axes[i, j]
            r = cells[cell]["results"]
            M = np.full((2, 3), np.nan)
            for a, (_, rk) in enumerate(DEC_ROWS):
                for b, (_, ck) in enumerate(DEC_COLS):
                    v = r[f"{rk}|{ck}"]["LaECE_0"]
                    M[a, b] = np.nan if v is None else v * 100
            ax.imshow(np.nan_to_num(M, nan=0.0), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
            for a in range(2):
                for b in range(3):
                    txt = "n/a" if np.isnan(M[a, b]) else f"{M[a, b]:.1f}"
                    ax.text(b, a, txt, ha="center", va="center", fontsize=7.5, color="#ffffff" if M[a, b] > 0.5 * vmax else INK)
            d_thr = M[0, 0] - M[1, 0]      # move along the threshold axis (target threshold, no map)
            d_map = M[0, 0] - M[0, 1]      # move along the map axis (source-fitted Platt, source threshold)
            run, tgt = cell.split("__to__")
            model, src = RUN_LABEL[run].split(" / ")
            dl = f"Δthr {-d_thr:+.1f} · Δmap {-d_map:+.1f}".replace("-", "−")
            ax.set_title(f"{model}\n{src} → {TARGET_LABEL[tgt]}\n{dl}", fontsize=6.0, loc="left", color=INK, pad=3, linespacing=1.25)
            ax.set_xticks(range(3)); ax.set_yticks(range(2))
            ax.set_xticklabels([c[0] for c in DEC_COLS] if i == 1 else [], fontsize=6.0)
            ax.set_yticklabels([r_[0] for r_ in DEC_ROWS] if j == 0 else [], fontsize=6.3)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            kv[cell] = {"thr=source|map=none": round(M[0, 0], 1), "thr=source|map=source:platt": round(M[0, 1], 1),
                        "thr=source|map=target:platt": round(M[0, 2], 1), "thr=target|map=none": round(M[1, 0], 1),
                        "thr=target|map=source:platt": round(M[1, 1], 1), "thr=target|map=target:platt": round(M[1, 2], 1),
                        "delta_threshold_axis": round(d_thr, 1), "delta_map_axis_source_platt": round(d_map, 1)}
    axes[1, 0].set_ylabel("threshold origin", fontsize=7)
    axes[1, 2].set_xlabel("confidence map", fontsize=7)
    fig.suptitle("LaECE (%) on the target test split: threshold origin × confidence map, seed 3407 core cells", x=0.125, ha="left", fontsize=9, color=INK, y=1.0)
    fig.text(0.125, -0.005, "Δthr = change in LaECE from re-selecting the threshold on the target (no map);  "
             "Δmap = change from a source-fitted Platt map at the source threshold.",
             fontsize=6.2, color=INK2, ha="left", va="top")
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.018, pad=0.03, aspect=30)
    cb.set_label("LaECE (%)", fontsize=7); cb.ax.tick_params(labelsize=6.5, color=GRID); cb.outline.set_visible(False)
    save(fig, "fig2_decomposition_core", ["threshold_decomposition.json: cells.<cell>.results.<thr=...|map=...>.LaECE_0 (x100)"], kv)


# ----------------------------------------------------------------------------- fig5 core
CORE_CI = [  # forward (source-fitted) cells of the six core seed-3407 models, grouped
    ("RT-DETR-l / D-Fire", ["rtdetrl_dfire_s3407__to__d_fire_test", "rtdetrl_dfire_s3407__to__pyro_sdis_caltest", "rtdetrl_dfire_s3407__to__thesis_test"]),
    ("YOLOv8-s / D-Fire", ["v8s_dfire_s3407__to__d_fire_test", "v8s_dfire_s3407__to__pyro_sdis_caltest", "v8s_dfire_s3407__to__thesis_test"]),
    ("YOLO11-s / D-Fire", ["y11s_dfire_s3407__to__d_fire_test", "y11s_dfire_s3407__to__pyro_sdis_caltest", "y11s_dfire_s3407__to__thesis_test"]),
    ("Pyro-SDIS-trained", ["v8s_pyrosdis_s3407__to__d_fire_test_smokeonly", "y11s_pyrosdis_s3407__to__d_fire_test_smokeonly"]),
]
CALS = [("temperature_scaling", "temperature", ORD[0], "o"), ("platt_scaling", "Platt", ORD[2], "s"), ("isotonic_regression", "isotonic", ORD[4], "^")]


def fig5_core(ci_dir):
    rows, ylab, kv = [], [], {}
    y = 0
    for gname, cells in CORE_CI:
        for c in cells:
            d = json.loads((ci_dir / f"{c}.json").read_text(encoding="utf-8"))
            run, tgt = c.split("__to__")
            in_dom = ("_dfire_" in run and tgt.startswith("d_fire")) or ("_pyrosdis_" in run and tgt.startswith("pyro"))
            lab = (RUN_LABEL[run].split(" / ")[0] if gname != "Pyro-SDIS-trained" else RUN_LABEL[run]) + f" → {TARGET_LABEL[tgt]}" + (" (in-domain)" if in_dom else "")
            rows.append((y, c, d["paired_vs_identity"])); ylab.append(lab)
            kv[c] = {k: {"point_diff_pts": round(d["paired_vs_identity"][k]["LaECE_0"]["point_diff"] * 100, 2),
                         "ci95_pts": [round(v * 100, 2) for v in d["paired_vs_identity"][k]["LaECE_0"]["ci95"]]} for k, _, _, _ in CALS}
            y += 1
        y += 1.0   # group gap (heading sits in it)
    fig, ax = plt.subplots(figsize=(3.5, 5.3))
    for yy, c, pv in rows:
        for k, (key, lab, col, mk) in enumerate(CALS):
            e = pv[key]["LaECE_0"]; off = (k - 1) * 0.26
            ax.plot([e["ci95"][0] * 100, e["ci95"][1] * 100], [yy + off, yy + off], color=col, linewidth=1.5, solid_capstyle="round", zorder=2)
            ax.scatter([e["point_diff"] * 100], [yy + off], color=col, marker=mk, s=16, zorder=3, edgecolors="#ffffff", linewidths=0.6,
                       label=lab if yy == 0 else None)
    ax.axvline(0, color=INK2, linewidth=0.8, zorder=1)
    # group brackets / headings
    gy = 0
    x0 = ax.get_xlim()[0]
    for gname, cells in CORE_CI:
        ax.text(x0, gy - 0.62, gname, fontsize=6.5, color=INK3, va="center", ha="left", clip_on=False)
        gy += len(cells) + 1.0
    ax.set_ylim(gy - 1.5, -1.0)   # inverted: first row at the top, headings inside the group gaps
    ax.set_yticks([r[0] for r in rows]); ax.set_yticklabels(ylab, fontsize=6.8)
    ax.set_xlabel("Δ LaECE (points), calibrator − identity; 95 % paired image-bootstrap CI\n(negative = post-hoc calibration helps)", fontsize=7)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.35, -0.14), ncol=3, title="calibrator (fitted on the source calibration split)", title_fontsize=6.5)
    ax.set_title("Source-fitted post-hoc calibration, paired effect", loc="left", color=INK)
    save(fig, "fig5_paired_posthoc_core", ["calibration_ci/<cell>.json: paired_vs_identity.<cal>.LaECE_0.{point_diff,ci95} (x100)"], kv)


# ----------------------------------------------------------------------------- fig8 label-free
QQ_CELLS = [("v8s_dfire_s3407__to__pyro_sdis_caltest", "YOLOv8-s / D-Fire → Pyro-SDIS", FAM["v8"][1], "o"),
            ("rtdetrl_dfire_s3407__to__pyro_sdis_caltest", "RT-DETR-l / D-Fire → Pyro-SDIS", FAM["rtdetr"][1], "s")]
QQ_Q = np.linspace(0.01, 0.99, 99)


def tp_scores(run, tag, classes):
    """Matched true-positive scores at IoU 0.5 on <tag> (same matcher/classes as the pilot)."""
    ids, boxes, _ = pilot.op.load_gt(tag)
    dets = pilot.op.load_dets(pilot.DETS / f"{run}__{tag}.bbox.json")
    sc, tp, n_gt, _ = pilot.op.match(ids, boxes, dets, classes)
    return sc[tp], int(n_gt), int(len(sc))


def fig8(refine, pil):
    ref_cells, pil_cells = refine["cells"], pil["cells"]
    fig = plt.figure(figsize=(7.2, 5.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.62, wspace=0.32)
    kv = {}
    # ---- (a) 44 cells ---------------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    names = sorted(ref_cells, key=lambda c: pil_cells[c]["shift_diagnostics"]["wasserstein_imgmax"])
    W = np.array([pil_cells[c]["shift_diagnostics"]["wasserstein_imgmax"] for c in names])
    F = {m: np.array([ref_cells[c]["methods"][m]["f1"] for c in names]) for m in ("M0_t_src", "M1c_kept_rate", "ORC_target_calval")}
    fams = [ref_cells[c]["family"] for c in names]
    x = np.arange(len(names))
    for i in x:
        col = FAM["rtdetr"][1] if fams[i] == "rtdetr" else FAM["v8"][1]; mk = "s" if fams[i] == "rtdetr" else "o"
        ax.plot([i, i], [F["M0_t_src"][i], F["ORC_target_calval"][i]], color=GRID, linewidth=1.0, zorder=1)
        ax.scatter([i], [F["ORC_target_calval"][i]], marker="_", s=70, color=INK, linewidths=1.3, zorder=3)
        ax.scatter([i], [F["M0_t_src"][i]], marker=mk, s=22, facecolors="#ffffff", edgecolors=col, linewidths=1.1, zorder=4)
        ax.scatter([i], [F["M1c_kept_rate"][i]], marker=mk, s=22, facecolors=col, edgecolors="#ffffff", linewidths=0.6, zorder=5)
    # categorical axis: cells are equally spaced, so ticks show the cell RANK; W_1 is annotated in a strip under the axis at every 5th cell
    ticks = list(range(0, len(names), 5))
    if ticks[-1] != len(names) - 1:
        ticks.append(len(names) - 1)
    ax.set_xticks(ticks); ax.set_xticklabels([str(t + 1) for t in ticks], fontsize=6.5); ax.tick_params(axis="x", length=2, color=GRID)
    ax.set_xlim(-1, len(names)); ax.set_ylim(0, max(0.62, F["ORC_target_calval"].max() + 0.03))
    ax.set_xlabel("44 cross-dataset cells (rank), ordered by shift diagnostic $W_1$ of the per-image max score, source vs target calibration split",
                  fontsize=7, labelpad=14)
    ax.text(-1, -0.135, "$W_1$", transform=ax.get_xaxis_transform(), ha="right", va="center", fontsize=6.3, color=INK2)
    for t in ticks:
        ax.text(t, -0.135, f"{W[t]:.2f}", transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=6.3, color=INK2)
    ax.set_ylabel("box F1 on the target test", fontsize=7)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    g = refine["summary"]["groups"]
    m0, m1, orc = g["all"]["M0_t_src"]["f1_mean"], g["all"]["M1c_kept_rate"]["f1_mean"], g["all"]["ORC_target_calval"]["f1_mean"]
    r_m0, r_m1 = g["all"]["M0_t_src"]["regret_mean"], g["all"]["M1c_kept_rate"]["regret_mean"]
    ax.text(0.01, 0.97, f"mean F1 over 44 cells: shipped {m0:.3f} → kept-rate rule {m1:.3f} (oracle {orc:.3f});\n"
                        f"mean regret vs oracle {r_m0:.3f} → {r_m1:.3f}", transform=ax.transAxes, fontsize=6.8, va="top", color=INK2)
    h = [Line2D([], [], marker="o", markerfacecolor="#ffffff", markeredgecolor=INK2, linestyle="", label="shipped threshold $t_{src}$ (open)"),
         Line2D([], [], marker="o", markerfacecolor=INK2, markeredgecolor="#ffffff", linestyle="", label="kept-rate rule, label-free (filled)"),
         Line2D([], [], marker="_", color=INK, markersize=9, markeredgewidth=1.3, linestyle="", label="labelled oracle (bar)"),
         Line2D([], [], marker="o", color=FAM["v8"][1], linestyle="", label="YOLO (circle)"),
         Line2D([], [], marker="s", color=FAM["rtdetr"][1], linestyle="", label="RT-DETR (square)")]
    ax.legend(handles=h, frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=5, handletextpad=0.3, columnspacing=1.0)
    ax.set_title("a. Threshold transfer per cell: shipped vs label-free vs oracle", loc="left", color=INK)
    kv["a"] = {"n_cells": len(names), "W1_imgmax_min_max": [round(float(W.min()), 3), round(float(W.max()), 3)],
               "f1_mean_all": {"M0_t_src": round(m0, 3), "M1c_kept_rate": round(m1, 3), "ORC_target_calval": round(orc, 3)},
               "regret_mean_all": {"M0_t_src": round(r_m0, 4), "M1c_kept_rate": round(r_m1, 4)},
               "n_cells_kept_rate_beats_t_src": g["all"]["M1c_kept_rate"].get("n_beats_t_src"),
               "n_cells_kept_rate_worse_than_t_src": g["all"]["M1c_kept_rate"].get("n_worse_than_t_src"),
               "order": names, "family_order": fams}
    # ---- (b) unlabelled data size curve ----------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    curve = refine["summary"]["unlabelled_data_curve"]
    ns = sorted(int(n) for n in curve)
    gcol = {"all": INK2, "yolo": FAM["v8"][1], "rtdetr": FAM["rtdetr"][1]}
    glab = {"all": "all 44 cells", "yolo": "YOLO (34)", "rtdetr": "RT-DETR (10)"}
    kv["b"] = {}
    for grp in ("all", "yolo", "rtdetr"):
        mean = [curve[str(n)]["M1c_kept_rate"][grp]["regret_mean"] for n in ns]
        p90 = [curve[str(n)]["M1c_kept_rate"][grp]["regret_p90"] for n in ns]
        ncell = [curve[str(n)]["M1c_kept_rate"][grp]["n_cells"] for n in ns]
        full_n = [n for n, k in zip(ns, ncell) if k == ncell[0]]          # n at which every cell of the group is present
        part_n = [n for n, k in zip(ns, ncell) if k != ncell[0]]          # n at which cells with a smaller calval drop out
        kf = len(full_n)
        ax.plot(ns[:kf], mean[:kf], "-", color=gcol[grp], linewidth=1.6, marker="o", markersize=3.5, label=f"{glab[grp]}, mean", zorder=3)
        ax.plot(ns[:kf], p90[:kf], "--", color=gcol[grp], linewidth=1.1, marker="o", markersize=2.5, markerfacecolor="#ffffff", label=f"{glab[grp]}, 90th pct.", zorder=2)
        if part_n:   # subset of cells: detached, open markers
            ax.scatter(part_n, mean[kf:], marker="o", s=14, facecolors="#ffffff", edgecolors=gcol[grp], linewidths=1.2, zorder=3)
            ax.scatter(part_n, p90[kf:], marker="o", s=8, facecolors="#ffffff", edgecolors=gcol[grp], linewidths=0.8, zorder=2)
        kv["b"][grp] = {"n": ns, "regret_mean": [round(v, 4) for v in mean], "regret_p90": [round(v, 4) for v in p90], "n_cells": ncell}
    full = g["all"]["M1c_kept_rate"]["regret_mean"]
    ax.axhline(full, color=INK2, linewidth=0.7, linestyle=":", zorder=1)
    n_last = kv["b"]["all"]["n_cells"][-1]
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns], fontsize=6.5); ax.minorticks_off()
    ax.set_xlabel("unlabelled target images used by the kept-rate rule\n(10 draws per cell)", fontsize=7)
    ax.set_ylabel("regret in F1 vs labelled oracle", fontsize=7); ax.set_ylim(0, None)
    ax.text(ns[-1] * 1.12, ax.get_ylim()[1], f"open, detached: n = {ns[-1]} keeps only\nthe {n_last} cells whose calibration split has\n≥ {ns[-1]} images (thesis calibration split = 590)",
            fontsize=5.6, color=INK2, ha="right", va="top")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color=INK2, linewidth=0.7, linestyle=":")); l.append(f"full adaptation set, all cells ({full:.4f})")
    ax.legend(h, l, frameon=False, fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.30), handlelength=2.2)
    kv["b"]["full_calval_regret_mean_all"] = round(full, 4)
    ax.set_title("b. Regret vs size of the unlabelled adaptation set", loc="left", color=INK)
    # ---- (c) QQ of true-detection scores ---------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    lo, hi = logit(pilot.EPS), logit(1 - pilot.EPS)
    ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=0.8, zorder=1)
    kv["c"] = {}
    for cell, lab, col, mk in QQ_CELLS:
        pc = pil_cells[cell]; run = pc["run"]
        src_tag, cal_tag = pc["source_calval"], pc["target_calval_unlabelled"]
        _, _, s_cls = pilot.op.load_gt(src_tag)
        cls_s = set(pc["classes_source"]) & s_cls
        cls_t = set(pc["classes_evaluated"])
        s_tp, s_ngt, s_nd = tp_scores(run, src_tag, cls_s)
        t_tp, t_ngt, t_nd = tp_scores(run, cal_tag, cls_t)
        qs, qt = np.quantile(s_tp, QQ_Q), np.quantile(t_tp, QQ_Q)
        xs, ys = logit(np.clip(qs, pilot.EPS, 1 - pilot.EPS)), logit(np.clip(qt, pilot.EPS, 1 - pilot.EPS))
        ax.scatter(xs, ys, marker=mk, s=11, color=col, edgecolors="#ffffff", linewidths=0.4, zorder=3, label=lab)
        m3 = pc["mixture"].get("beta_mle_shift")
        delta = m3["delta_logit"] if m3 else None
        if delta is not None:
            ax.plot([lo, hi], [lo + delta, hi + delta], "--", color=col, linewidth=1.0, zorder=2)
        med_s, med_t = float(np.median(s_tp)), float(np.median(t_tp))
        kv["c"][cell] = {"n_tp_source_calval": int(len(s_tp)), "n_gt_source": s_ngt, "n_dets_source": s_nd,
                         "n_tp_target_calval": int(len(t_tp)), "n_gt_target": t_ngt, "n_dets_target": t_nd,
                         "median_tp_score_source": round(med_s, 3), "median_tp_score_target": round(med_t, 3),
                         "median_logit_shift_observed": round(float(logit(med_t) - logit(med_s)), 3),
                         "M3_beta_mle_shift_delta_logit": delta, "M3_delta_at_grid_edge": m3.get("delta_at_grid_edge") if m3 else None,
                         "classes_source": sorted(cls_s), "classes_target": sorted(cls_t)}
    tickv = [0.01, 0.1, 0.5, 0.9, 0.99]
    ax.set_xticks(logit(tickv)); ax.set_xticklabels([str(t) for t in tickv], fontsize=6.5)
    ax.set_yticks(logit(tickv)); ax.set_yticklabels([str(t) for t in tickv], fontsize=6.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("true-detection score quantile,\nsource calibration split (logit axis)", fontsize=7)
    ax.set_ylabel("same quantile,\ntarget calibration split", fontsize=7)
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    txt = "dashed = shared logit shift fitted by the prior-shift mixture\nmedian true-detection score, source → target:\n" + "\n".join(
        f"{lab.split(' / ')[0]}: $\\hat\\delta$ = {kv['c'][cell]['M3_beta_mle_shift_delta_logit']:+.2f}; "
        f"{kv['c'][cell]['median_tp_score_source']:.2f} → {kv['c'][cell]['median_tp_score_target']:.2f}"
        for cell, lab, _, _ in QQ_CELLS)
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=6.0, va="top", color=INK2, linespacing=1.3)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_title("c. Where the true-detection scores go under shift", loc="left", color=INK)
    save(fig, "fig8_labelfree",
         ["label_free_threshold_refine.json: cells.<cell>.methods.{M0_t_src,M1c_kept_rate,ORC_target_calval}.f1; summary.groups.all.*.{f1_mean,regret_mean}",
          "label_free_threshold_pilot.json: cells.<cell>.shift_diagnostics.wasserstein_imgmax (ordering); cells.<cell>.mixture.beta_mle_shift.delta_logit (panel c)",
          "label_free_threshold_refine.json: summary.unlabelled_data_curve.<n>.M1c_kept_rate.{all,yolo,rtdetr}.{regret_mean,regret_p90}",
          "panel c recomputed: detections/<run>__{d_fire_calval,pyro_sdis_calval}.bbox.json + coco_gt/*.json via operating_points.match (IoU 0.5), TP scores, 1-99 % quantiles"],
         kv)


# ----------------------------------------------------------------------------- fig9 reliability
REL_RUNS = [("v8s_dfire_s3407", "YOLOv8-s"), ("rtdetrl_dfire_s3407", "RT-DETR-l")]
REL_TGTS = [("d_fire_test", "D-Fire test (in-domain)"), ("pyro_sdis_caltest", "Pyro-SDIS (shifted)")]
REL_BINS = np.linspace(0.0, 1.0, 26)
CAT_NAME = {1: "fire", 2: "smoke"}
CAT_STYLE = {1: (INK3, "--"), 2: (INK, "--")}


def _laece_bins(scores, tps, fps, ious):
    """Per-class LaECE_0 bin statistics exactly as the toolbox's compute_single_errors: 25 equal-width bins,
    bin 1 closed at both ends and the rest (lo, hi]; bin precision = sum of matched IoU / n_det; bin error =
    |precision - mean score|; bin weight = n_det / total. Returns (precision, mean score, weight, count, total)."""
    valid = tps | fps
    tot = int(valid.sum())
    prec, ms, w, n = np.full(25, np.nan), np.full(25, np.nan), np.zeros(25), np.zeros(25, int)
    for i in range(25):
        inb = (REL_BINS[i] <= scores) & (scores <= REL_BINS[i + 1]) if i == 0 else (REL_BINS[i] < scores) & (scores <= REL_BINS[i + 1])
        d = inb & valid
        n[i] = int(d.sum())
        if n[i] == 0:
            continue
        prec[i] = ious[inb & tps].sum() / n[i]
        ms[i] = scores[d].mean()
        w[i] = n[i] / tot
    return prec, ms, w, n, tot


def _laece_iou05_pooled(kept, tgt):
    """Reconciliation only: the IoU >= 0.5 greedy matcher of operating_points (pooled over classes, TP weighted by
    its IoU) binned the same way. NOT the artefact definition (LaECE_0 = toolbox matching at tau = 0, per class)."""
    ids, gtb, gcls = pilot.op.load_gt(tgt)
    byimg = {}
    for x in kept:
        byimg.setdefault(x["image_id"], []).append((x["category_id"], x["bbox"], x["score"]))
    sc, io = [], []
    for iid in ids:
        g = [(c, b) for c, b in gtb.get(iid, []) if c in gcls]
        used = [False] * len(g)
        for c, b, s in sorted([x for x in byimg.get(iid, []) if x[0] in gcls], key=lambda x: -x[2]):
            best, bj = 0.0, -1
            for j, (gc, gb) in enumerate(g):
                if used[j] or gc != c:
                    continue
                v = pilot.op.iou_xywh(b, gb)
                if v > best:
                    best, bj = v, j
            if best >= pilot.op.IOU:
                used[bj] = True
            sc.append(s); io.append(best if best >= pilot.op.IOU else 0.0)
    sc, io = np.array(sc), np.array(io)
    e = 0.0
    for i in range(25):
        inb = (REL_BINS[i] <= sc) & (sc <= REL_BINS[i + 1]) if i == 0 else (REL_BINS[i] < sc) & (sc <= REL_BINS[i + 1])
        if inb.sum():
            e += inb.sum() / len(sc) * abs(io[inb].mean() - sc[inb].mean())
    return float(e), int(len(sc))


def fig9_reliability(summary, decomp):
    import contextlib, io
    from detection_calibration.coco_calibration import CalibrationCOCO
    from detection_calibration.DetectionCalibration import DetectionCalibration
    from detection_calibration.utils import threshold_detections
    from pycocotools.coco import COCO

    def quiet(fn, *a, **k):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **k)

    val_gt = str(HERE / "coco_gt" / "d_fire_calval.json")
    dcls = list(COCO(val_gt).cats.keys())          # toolbox class order for the per-class thresholds (sorted ids)
    fig = plt.figure(figsize=(7.2, 7.0))
    outer = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.22, left=0.08, right=0.99, top=0.95, bottom=0.17)
    kv = {}
    xg = np.linspace(0.001, 0.999, 400)
    for i, (run, model) in enumerate(REL_RUNS):
        cal_art = {t: json.loads((HERE / "calibration" / f"{run}__to__{t}.json").read_text(encoding="utf-8")) for t, _ in REL_TGTS}
        # source LRP-optimal thresholds: threshold_decomposition (cross cell) must equal calibration/<cell>.json identity thresholds
        thr_raw = decomp[f"{run}__to__pyro_sdis_caltest"]["results"]["thr=source|map=none"]["thr_raw"]
        thr = np.array([thr_raw[str(c)] for c in dcls])
        for t, _ in REL_TGTS:
            assert np.allclose(thr, cal_art[t]["results"]["identity"]["thresholds"]["operating"]), (run, t)
        # source-fitted Platt map: refit with the toolbox exactly as run_calibration.py (fit on the D-Fire calval);
        # verified below against the operating thresholds stored in calibration/<cell>.json
        cm = DetectionCalibration(val_gt, str(HERE / "coco_gt" / "d_fire_test.json"))
        pmodel, pthr = quiet(cm.fit, str(HERE / "detections" / f"{run}__d_fire_calval.bbox.json"), calibrator_type="platt_scaling")
        stored = cal_art["d_fire_test"]["results"]["platt_scaling"]["thresholds"]
        assert np.allclose(np.atleast_1d(pthr[0]), stored["pre"]) and np.allclose(np.atleast_1d(pthr[1]), stored["operating"], atol=1e-6), (run, pthr, stored)
        platt = {int(dcls[k]): {"a_scale": float(abs(m.scale.detach()).item()), "b_shift": float(m.shift.detach().item())} for k, m in pmodel.items()}
        for j, (tgt, tlab) in enumerate(REL_TGTS):
            cell = f"{run}__to__{tgt}"
            test_gt = str(HERE / "coco_gt" / f"{tgt}.json")
            dets = json.loads((HERE / "detections" / f"{run}__{tgt}.bbox.json").read_text(encoding="utf-8"))
            kept = threshold_detections(threshold_detections(dets, thr, dcls), thr, dcls)   # identity transform: pre + operating
            ev = CalibrationCOCO(test_gt, test_gt, "bbox", 25, 0.0, False, False, 100)      # = calib_metrics.evaluate, first evaluator
            ev.cocoDt = quiet(COCO(test_gt).loadRes, kept)
            quiet(ev.evaluate); quiet(ev.prepare_input); quiet(ev.compute_single_errors)
            la_toolbox = float(ev.accumulate_errors())
            per_cls, cls_err = {}, []
            for k, ci in ev.calibration_info.items():
                if "tps" not in ci:
                    continue
                prec, ms, w, n, tot = _laece_bins(ci["scores"], ci["tps"], ci["fps"], ci["iou"])
                assert np.allclose(np.nan_to_num(prec), np.nan_to_num(ev.prec_iou[k])) and np.allclose(w, ev.weights_per_bin[k])
                e_cls = float(np.nansum(w * np.abs(prec - ms)))
                cls_err.append(e_cls if e_cls != 0 else np.nan)
                per_cls[int(ev.params.catIds[k])] = {"precision_iou": prec, "mean_score": ms, "weight": w, "count": n, "n_det": tot,
                                                    "n_tp": int(ci["tps"].sum()), "n_fp": int(ci["fps"].sum()), "LaECE_0_class": e_cls}
            la_drawn = float(np.nanmean(cls_err))
            la_matrix = summary[cell]["metrics"]["identity"]["LaECE_0"]
            assert abs(la_drawn - la_matrix) < 0.005 and abs(la_toolbox - la_matrix) < 1e-9, (cell, la_drawn, la_toolbox, la_matrix)
            la_iou05, n_iou05 = _laece_iou05_pooled(kept, tgt)
            print(f"[fig9] {cell}: LaECE_0 drawn {la_drawn:.5f} | toolbox rerun {la_toolbox:.5f} | matrix_summary {la_matrix:.5f} | "
                  f"IoU-0.5 pooled variant {la_iou05:.5f}; kept {len(kept)}")
            # class-averaged bins (the toolbox's own reliability diagram, cl = -1) --------------------------------
            P = np.array([v["precision_iou"] for v in per_cls.values()]); M = np.array([v["mean_score"] for v in per_cls.values()])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN bins (empty in every class)
                bar = np.nanmean(P, axis=0); dot = np.nanmean(M, axis=0)
            cnt = np.sum([v["count"] for v in per_cls.values()], axis=0)
            inner = outer[i, j].subgridspec(2, 1, height_ratios=[1, 0.32], hspace=0.10)
            ax = fig.add_subplot(inner[0]); axh = fig.add_subplot(inner[1], sharex=ax)
            col = FAM[fam_of(run)][1]
            ctr = (REL_BINS[:-1] + REL_BINS[1:]) / 2
            ok = ~np.isnan(bar)
            ax.bar(ctr[ok], bar[ok], width=0.04 * 0.9, color=col, alpha=0.75, linewidth=0, zorder=2, label="IoU-weighted precision of the bin (identity map)")
            ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.8, zorder=3, label="perfect calibration (identity)")
            ax.scatter(dot[ok], bar[ok], s=7, color=INK, zorder=5, label="bin mean confidence vs its precision (the gap LaECE sums)")
            for c in per_cls:
                a, b = platt[c]["a_scale"], platt[c]["b_shift"]
                lc, ls = CAT_STYLE[c]
                ax.plot(xg, expit(a * logit(xg) + b), ls, color=lc, linewidth=1.0, zorder=4, label=f"source-fitted Platt map, {CAT_NAME[c]} class")
                ax.axvline(thr[dcls.index(c)], color=lc, linewidth=0.6, linestyle=":", zorder=1,
                           label=f"source LRP-optimal threshold, {CAT_NAME[c]} class" if (i, j) == (0, 0) else None)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.tick_params(labelsize=7, labelbottom=False); ax.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
            letter = "abcd"[2 * i + j]
            ax.set_title(f"{letter}. {model} → {tlab}\nLaECE {la_drawn * 100:.1f} %  ·  {len(kept):,} detections above the source threshold",
                         loc="left", fontsize=8, color=INK, linespacing=1.3)
            if j == 0:
                ax.set_ylabel("IoU-weighted precision", fontsize=7); axh.set_ylabel("count", fontsize=7)
            axh.bar(ctr, np.maximum(cnt, 0), width=0.04 * 0.9, color=col, alpha=0.75, linewidth=0, zorder=2)
            axh.set_yscale("log"); axh.set_ylim(0.8, max(10, cnt.max() * 3)); axh.tick_params(labelsize=7); axh.minorticks_off()
            axh.yaxis.grid(True, color=GRID, linewidth=0.5); axh.set_axisbelow(True)
            axh.set_xlabel("predicted confidence, 25 equal-width bins", fontsize=7)
            kv[cell] = {"run": run, "target": tgt, "source_thresholds_by_class": {CAT_NAME[c]: float(thr[dcls.index(c)]) for c in dcls},
                        "n_detections_kept": int(len(kept)), "n_evaluated_by_class": {CAT_NAME[c]: v["n_det"] for c, v in per_cls.items()},
                        "LaECE_0_drawn_bins": round(la_drawn, 6), "LaECE_0_toolbox_rerun": round(la_toolbox, 6), "LaECE_0_matrix_summary": round(la_matrix, 6),
                        "LaECE_0_iou05_pooled_reconciliation_only": round(la_iou05, 6), "n_iou05_matcher": n_iou05,
                        "bars_class_mean_precision_iou": [None if np.isnan(v) else round(float(v), 4) for v in bar],
                        "dots_class_mean_score": [None if np.isnan(v) else round(float(v), 4) for v in dot],
                        "counts_per_bin": [int(v) for v in cnt],
                        "per_class": {CAT_NAME[c]: {"precision_iou": [None if np.isnan(x) else round(float(x), 4) for x in v["precision_iou"]],
                                                    "mean_score": [None if np.isnan(x) else round(float(x), 4) for x in v["mean_score"]],
                                                    "weight": [round(float(x), 5) for x in v["weight"]], "count": [int(x) for x in v["count"]],
                                                    "n_det": v["n_det"], "n_tp_tau0": v["n_tp"], "n_fp_tau0": v["n_fp"], "LaECE_0_class": round(v["LaECE_0_class"], 6)}
                                      for c, v in per_cls.items()},
                        "platt_source_fitted": {CAT_NAME[c]: platt[c] for c in per_cls},
                        "platt_refit_check": {"refit_operating_thresholds": [float(v) for v in np.atleast_1d(pthr[1])], "stored_operating_thresholds": stored["operating"]}}
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, handlelength=2.2, columnspacing=1.5)
    save(fig, "fig9_reliability",
         ["detections/<run>__<split>.bbox.json + coco_gt/<split>.json, thresholded at the source LRP-optimal thresholds "
          "(threshold_decomposition.json cells.<run>__to__pyro_sdis_caltest.results.thr=source|map=none.thr_raw == "
          "calibration/<cell>.json results.identity.thresholds.operating) and binned with the pinned toolbox CalibrationCOCO "
          "(bbox, 25 bins, tau = 0, maxDets 100) exactly as calib_metrics.evaluate; bars = class mean of the per-class bin IoU-precision",
          "matrix_summary.json: cells.<cell>.metrics.identity.LaECE_0 (asserted equal to the drawn-bin value within 0.005; toolbox rerun equal to 1e-9)",
          "Platt map: refitted with DetectionCalibration.fit(<run>__d_fire_calval.bbox.json, platt_scaling) as run_calibration.py does "
          "(parameters are not stored in the artefacts); refit verified against calibration/<cell>.json results.platt_scaling.thresholds.operating",
          "reconciliation only: LaECE_0_iou05_pooled = operating_points-style IoU >= 0.5 greedy matcher, pooled classes (NOT the artefact definition)"],
         kv)


def main():
    fig2_core(json.loads((HERE / "threshold_decomposition.json").read_text(encoding="utf-8")))
    fig5_core(HERE / "calibration_ci")
    fig8(json.loads((HERE / "label_free_threshold_refine.json").read_text(encoding="utf-8")),
         json.loads((HERE / "label_free_threshold_pilot.json").read_text(encoding="utf-8")))
    fig9_reliability(json.loads((HERE / "matrix_summary.json").read_text(encoding="utf-8"))["cells"],
                     json.loads((HERE / "threshold_decomposition.json").read_text(encoding="utf-8"))["cells"])
    (OUT / "figures_index_eaai.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print("index ->", OUT / "figures_index_eaai.json")


if __name__ == "__main__":
    main()
