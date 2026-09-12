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
                            with the pilot's M3 logit shift.
Sources: threshold_decomposition.json, calibration_ci/<cell>.json, label_free_threshold_refine.json,
label_free_threshold_pilot.json, detections/*.bbox.json + coco_gt/*.json (panel c, recomputed here
with operating_points.match via label_free_threshold_pilot).
"""
import json
import sys
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
DEC_COLS = [("none", "map=none"), ("src\nPlatt", "map=source:platt"), ("tgt\nPlatt", "map=target:platt")]


def fig2_core(decomp):
    cells = decomp["cells"]
    fig, axes = plt.subplots(2, 5, figsize=(7.5, 3.9), gridspec_kw={"wspace": 0.42, "hspace": 0.85})
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
            ax.set_xticklabels([c[0] for c in DEC_COLS] if i == 1 else [], fontsize=6.3)
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
    fig.suptitle("LaECE$_0$ (%) on the target test split: threshold origin × confidence map, seed 3407 core cells", x=0.125, ha="left", fontsize=9, color=INK, y=1.0)
    fig.text(0.125, -0.005, "Δthr = change in LaECE$_0$ from re-selecting the threshold on the target (no map);  "
             "Δmap = change from a source-fitted Platt map at the source threshold.",
             fontsize=6.2, color=INK2, ha="left", va="top")
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.018, pad=0.03, aspect=30)
    cb.set_label("LaECE$_0$ (%)", fontsize=7); cb.ax.tick_params(labelsize=6.5, color=GRID); cb.outline.set_visible(False)
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
    ax.set_xlabel("Δ LaECE$_0$ (points), calibrator − identity; 95 % paired image-bootstrap CI\n(negative = post-hoc calibration helps)", fontsize=7)
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
    txt = "dashed = pilot M3 fitted logit shift $\\hat\\delta$\n" + "\n".join(
        f"{lab.split(' / ')[0]}: $\\hat\\delta$ = {kv['c'][cell]['M3_beta_mle_shift_delta_logit']:+.2f}; "
        f"median TP {kv['c'][cell]['median_tp_score_source']:.2f} → {kv['c'][cell]['median_tp_score_target']:.2f}"
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


def main():
    fig2_core(json.loads((HERE / "threshold_decomposition.json").read_text(encoding="utf-8")))
    fig5_core(HERE / "calibration_ci")
    fig8(json.loads((HERE / "label_free_threshold_refine.json").read_text(encoding="utf-8")),
         json.loads((HERE / "label_free_threshold_pilot.json").read_text(encoding="utf-8")))
    (OUT / "figures_index_eaai.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print("index ->", OUT / "figures_index_eaai.json")


if __name__ == "__main__":
    main()
