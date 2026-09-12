"""Paper figures from the machine-written artefacts (rerun after every refresh).

Static, print-first (MDPI single/double column), matplotlib. Palette = the
hub's validated reference palette (dataviz skill): categorical slots
blue #2a78d6 / orange #eb6834 / aqua #1baf7a for the three families (all-pairs
validated), sequential blue ramp for magnitude, text in ink tokens only.
Outputs analysis/figures_paper/<name>.{png,pdf} and a figures_index.json
listing the artefact each panel reads.
  fig1_transfer_matrix  identity LaECE_0 heatmap, runs x targets (+ CI text)
  fig2_decomposition    threshold-vs-map grid per cross cell (ordinal ramp)
  fig3_leakage          D-Fire leaky vs clean test: mAP50 and LaECE_0
  fig4_operating_points image-level sensitivity vs false-alarm at t_src
  fig5_paired_posthoc   forest plot of paired (calibrator - identity) effects
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (Type 42), not Type 3, for Elsevier
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures_paper"
OUT.mkdir(exist_ok=True)
FAM = {"v8": ("YOLOv8", "#2a78d6"), "y11": ("YOLO11", "#eb6834"), "rtdetr": ("RT-DETR", "#1baf7a")}
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
TARGET_LABEL = {"d_fire_test": "D-Fire test", "d_fire_test_dedup": "D-Fire clean", "d_fire_test_smokeonly": "D-Fire smoke",
                "d_fire_test_smokeonly_dedup": "D-Fire smoke clean", "pyro_sdis_caltest": "Pyro-SDIS", "thesis_test": "Thesis set",
                "thesis_test_smokeonly": "Thesis smoke", "figlib_bb": "FIgLib (boxes)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9, "axes.titleweight": "normal",
                     "figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
index = {}


def keep_cell(cell):
    """One row per (run, target): smoke-only sources use the *_smokeonly GT, others the full GT; no dedup/repair rows."""
    run, tgt = cell.split("__to__")
    if tgt.endswith("__repair") or "dedup" in tgt:
        return False
    smoke_src = "pyrosdis" in run
    if tgt.startswith("pyro_sdis"):
        return True
    return tgt.endswith("smokeonly") == smoke_src


def run_label(run):
    m = re.match(r"^(v8|y11|rtdetr)([nsmlx])_(dfire|pyrosdis|dfiredd|dfiresub)_s(\d+)(_[a-z0-9]+)?$", run)
    fam, size, src, seed, tag = m.groups()
    src_l = {"dfire": "D-Fire", "pyrosdis": "Pyro-SDIS", "dfiredd": "D-Fire de-duplicated", "dfiresub": "D-Fire leaky subset"}[src]
    return f"{FAM[fam][0]}-{size} / {src_l}" + (f" ({tag[1:]})" if tag else ""), fam


def save(fig, name, sources):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    index[name] = sources
    print("[fig]", name)


def fig1(summary):
    cells = summary["cells"]
    runs = sorted({c["run"] for c in cells.values() if c.get("kind") in ("in_domain", "cross") and c.get("seed") == 3407 and not c.get("variant")})
    targets = ["d_fire_test", "d_fire_test_dedup", "pyro_sdis_caltest", "thesis_test"]
    M = np.full((len(runs), len(targets)), np.nan); T = [["" for _ in targets] for _ in runs]
    dagger = []   # cells whose LRP-FN (false-negative component at the operating threshold) exceeds 0.95
    for i, r in enumerate(runs):
        for j, t in enumerate(targets):
            tt = t
            if "pyrosdis" in r and t.startswith("d_fire_test"):
                tt = t.replace("d_fire_test", "d_fire_test_smokeonly")
            c = cells.get(f"{r}__to__{tt}")
            if c:
                v = c["metrics"]["identity"]["LaECE_0"] * 100; M[i, j] = v
                ci = c.get("ci95", {}).get("identity", {}).get("LaECE_0")
                fn = c["metrics"]["identity"].get("lrp_fn")
                mark = fn is not None and fn > 0.95
                if mark:
                    dagger.append(f"{r}__to__{tt}")
                T[i][j] = f"{v:.1f}" + ("\u2020" if mark else "") + (f"\n[{ci[0]*100:.1f}, {ci[1]*100:.1f}]" if ci else "")
    fig, ax = plt.subplots(figsize=(3.4, 0.42 * len(runs) + 0.9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", SEQ)
    ax.imshow(np.nan_to_num(M, nan=0), cmap=cmap, vmin=0, vmax=max(45, np.nanmax(M)), aspect="auto")
    for i in range(len(runs)):
        for j in range(len(targets)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, T[i][j], ha="center", va="center", fontsize=6.5, color="#ffffff" if M[i, j] > 22 else INK)
    ax.set_xticks(range(len(targets))); ax.set_xticklabels([TARGET_LABEL[t] for t in targets])
    ax.set_yticks(range(len(runs))); ax.set_yticklabels([run_label(r)[0] for r in runs])
    ax.tick_params(length=0); ax.set_title("Native LaECE (%) with 95 % image-bootstrap CI", loc="left", color=INK)
    for s in ax.spines.values(): s.set_visible(False)
    ax.text(0, -0.03, "\u2020 LRP-FN > 0.95: the model detects almost nothing in this domain,\nso LaECE is largely the confidence of false positives",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK2, linespacing=1.3)
    print("[fig1] dagger cells (lrp_fn > 0.95):", dagger)
    save(fig, "fig1_transfer_matrix", ["matrix_summary.json: cells.<run>__to__<target>.metrics.identity.LaECE_0", "calibration_ci/*.json: results.identity.LaECE_0.ci95",
                                       "dagger marks (lrp_fn > 0.95): matrix_summary.json cells.<cell>.metrics.identity.lrp_fn -> " + ", ".join(dagger)])


def fig2(decomp):
    keys = ["thr=source|map=none", "thr=source|map=source:platt", "thr=source|map=target:platt", "thr=target|map=none", "thr=target|map=target:platt"]
    labels = ["source threshold, no map", "source threshold, source Platt", "source threshold, target Platt", "target threshold, no map", "target threshold, target Platt"]
    cells = [c for c in decomp["cells"] if keep_cell(c)]
    fig, ax = plt.subplots(figsize=(7.0, 0.32 * len(cells) + 1.6))
    y = np.arange(len(cells)); h = 0.15
    for k, (key, lab, col) in enumerate(zip(keys, labels, ORD)):
        vals = [(decomp["cells"][c]["results"][key]["LaECE_0"] or 0) * 100 for c in cells]
        ax.barh(y + (k - 2) * h, vals, height=h * 0.9, color=col, label=lab, linewidth=0)
    ax.set_yticks(y); ax.set_yticklabels([f"{run_label(c.split('__to__')[0])[0]} → {TARGET_LABEL.get(c.split('__to__')[1].replace('__repair',''), c.split('__to__')[1])}" for c in cells])
    ax.invert_yaxis(); ax.set_xlabel("LaECE (%) on the target test split"); ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.12), fontsize=7, title="threshold origin, confidence map", title_fontsize=7)
    ax.set_title("Where the miscalibration under shift comes from: operating threshold vs confidence map", loc="left", color=INK)
    save(fig, "fig2_decomposition", ["threshold_decomposition.json: cells.<cell>.results.<condition>.LaECE_0"])


def fig3(leak):
    cells = {c: v for c, v in leak["cells"].items() if "_dfire_" in c}
    runs = sorted(cells)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={"wspace": 0.28})
    x = np.arange(len(runs)); w = 0.36

    def tick(r):   # "RT-DETR-l, seed 3407": model + training seed, rotated 45 deg so ten labels do not over-print
        m = re.match(r"^(v8|y11|rtdetr)([nsmlx])_dfire_s(\d+)", r.split("__to__")[0])
        return f"{FAM[m.group(1)][0]}-{m.group(2)}, seed {m.group(3)}"
    ax = axes[0]
    ax.bar(x - w / 2, [cells[r]["mAP50_leaky"] for r in runs], w * 0.95, color=ORD[1], label="leaky test images (train near-duplicates)", linewidth=0)
    ax.bar(x + w / 2, [cells[r]["mAP50_clean"] for r in runs], w * 0.95, color=ORD[3], label="clean test images", linewidth=0)
    ax.set_ylim(0, 1); ax.set_ylabel("mAP50")
    ax.set_title("a. Accuracy on D-Fire test", loc="left", color=INK); ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=1)
    ax = axes[1]
    ax.bar(x - w / 2, [cells[r]["LaECE_0_full_test"]["identity"] * 100 for r in runs], w * 0.95, color=ORD[1], label="official test (leaky)", linewidth=0)
    ax.bar(x + w / 2, [cells[r]["LaECE_0"]["identity"] * 100 for r in runs], w * 0.95, color=ORD[3], label="clean test", linewidth=0)
    ax.set_ylabel("native LaECE (%)")
    ax.set_ylim(0, 20); ax.set_yticks([0, 5, 10, 15, 20]); ax.set_title("b. Calibration on D-Fire test", loc="left", color=INK); ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=1)
    for a in axes:
        a.yaxis.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True)
        a.set_xticks(x); a.set_xticklabels([tick(r) for r in runs], rotation=45, ha="right", rotation_mode="anchor", fontsize=6.5)
        a.tick_params(axis="x", length=0)
    save(fig, "fig3_leakage", ["dfire_leakage.json: cells.<run>__to__d_fire_test_dedup.{mAP50_leaky,mAP50_clean,LaECE_0,LaECE_0_full_test}"])


def fig4(ops):
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    markers = {"dfire": "o", "pyrosdis": "s", "thesis": "^"}
    seen = set()
    for cell, o in ops["cells"].items():
        if cell.endswith("__repair") or "dedup" in cell or cell.endswith("smokeonly"):
            continue
        run = o["run"]; lab, fam = run_label(run)
        tgt = "dfire" if o["target_test"].startswith("d_fire") else "pyrosdis" if o["target_test"].startswith("pyro") else "thesis"
        il = o["target"]["image_level_at_t_src"]
        if not il or il["sensitivity"] is None:
            continue
        in_domain = (tgt == "dfire" and "_dfire_" in run) or (tgt == "pyrosdis" and "_pyrosdis_" in run)
        ax.scatter(il["false_alarm_rate"], il["sensitivity"], s=42 if in_domain else 28, marker=markers[tgt], color=FAM[fam][1],
                   edgecolors="#ffffff", linewidths=1.2, zorder=3, label=FAM[fam][0] if fam not in seen else None)
        seen.add(fam)
    ax.set_xlabel("image-level false-alarm rate at shipped threshold"); ax.set_ylabel("image-level sensitivity")
    ax.set_xlim(-0.02, 0.55); ax.set_ylim(-0.02, 1.02); ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=INK2, linestyle="", label="target: D-Fire"), Line2D([], [], marker="s", color=INK2, linestyle="", label="target: Pyro-SDIS"),
         Line2D([], [], marker="^", color=INK2, linestyle="", label="target: thesis set")]
    leg1 = ax.legend(frameon=False, fontsize=6.5, loc="center", bbox_to_anchor=(0.66, 0.40), title="family", title_fontsize=6.5); ax.add_artist(leg1)
    ax.legend(handles=h, frameon=False, fontsize=6.5, loc="lower right")
    ax.annotate("Pyro-SDIS in-domain:\nnegatives are event-adjacent frames", xy=(0.47, 0.83), xytext=(0.14, 0.63), fontsize=6.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.6))
    ax.set_title("Alarm behaviour at the source-selected threshold", loc="left", color=INK)
    save(fig, "fig4_operating_points", ["operating_points.json: cells.<cell>.target.image_level_at_t_src.{sensitivity,false_alarm_rate}"])


def fig5(ci_dir):
    rows = []
    for p in sorted(ci_dir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8")); c = d["cell"]
        if not keep_cell(c):
            continue
        for cal, lab in (("platt_scaling", "Platt"), ("isotonic_regression", "isotonic")):
            e = d["paired_vs_identity"][cal]["LaECE_0"]
            rows.append((c, lab, e["point_diff"] * 100, e["ci95"][0] * 100, e["ci95"][1] * 100))
    cells = sorted({r[0] for r in rows}, key=lambda c: (0 if "rtdetr" in c else 1, c))
    fig, ax = plt.subplots(figsize=(4.2, 0.3 * len(cells) + 1.0))
    for i, c in enumerate(cells):
        for k, (lab, col) in enumerate((("Platt", ORD[2]), ("isotonic", ORD[4]))):
            r = next(x for x in rows if x[0] == c and x[1] == lab)
            yy = i + (k - 0.5) * 0.3
            ax.plot([r[3], r[4]], [yy, yy], color=col, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter([r[2]], [yy], color=col, s=18, zorder=3, edgecolors="#ffffff", linewidths=0.8, label=lab if i == 0 else None)
    ax.axvline(0, color=INK2, linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(cells))); ax.set_yticklabels([f"{run_label(c.split('__to__')[0])[0]} → {TARGET_LABEL.get(c.split('__to__')[1], c.split('__to__')[1])}" for c in cells]); ax.invert_yaxis()
    ax.set_xlabel("change in LaECE (points) from source-fitted post-hoc calibration, 95 % paired bootstrap CI")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("Source-fitted post-hoc calibration: paired effect vs no calibration", loc="left", color=INK)
    save(fig, "fig5_paired_posthoc", ["calibration_ci/<cell>.json: paired_vs_identity.<cal>.LaECE_0.{point_diff,ci95}"])


def fig6_figlib():
    """FIgLib: frame-level ROC (pre-plume FAR vs smoke-visible sensitivity) per model, plus
    median time-to-detection at matched FAR (camera-disjoint)."""
    gt = json.loads((HERE / "coco_gt" / "figlib_all.json").read_text(encoding="utf-8"))
    pos = np.array([im["smoke_visible"] for im in gt["images"]]); ids = [str(im["id"]) for im in gt["images"]]
    ttd = json.loads((HERE / "figlib_ttd.json").read_text(encoding="utf-8"))["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.55})
    ax = axes[0]
    def figlib_label(run):
        """'RT-DETR-l (Pyro-SDIS, patience 60)' / '(..., default protocol)' for the two RT-DETR Pyro-SDIS runs; 'model (source)' otherwise."""
        lab, fam = run_label(run)
        model, src = lab.split(" / ")
        if src.endswith(" (p60)"):
            return f"{model} ({src[:-6]}, patience 60)", fam
        if run.startswith("rtdetr") and "_pyrosdis_" in run:
            return f"{model} ({src}, default protocol)", fam
        return f"{model} ({src})", fam
    for sp in sorted((HERE / "figlib_scores").glob("*.json")):
        run = sp.stem
        if run not in ttd:
            continue
        lab, fam = figlib_label(run)
        raw = json.loads(sp.read_text(encoding="utf-8")); key = "max_smoke" if "_dfire_" in run else "max"
        sc = np.array([raw[i][key] for i in ids])
        ts = np.quantile(sc[~pos], 1 - np.linspace(0.001, 0.5, 120))
        far = [(sc[~pos] >= t).mean() for t in ts]; sens = [(sc[pos] >= t).mean() for t in ts]
        if "_dfire_" in run:
            ls, mk = "-", ""
        elif run.endswith("_p60"):
            ls, mk = (0, (1.2, 1.2)), "o"       # dotted + open circles: patience-60 protocol
        else:
            ls, mk = "--", ""                     # dashed: default protocol
        ax.plot(far, sens, linestyle=ls, marker=mk, markevery=12, markersize=3.2, markerfacecolor="#ffffff", markeredgewidth=1.0,
                color=FAM[fam][1], linewidth=1.6, label=lab)
    ax.set_xscale("log"); ax.set_xlim(0.001, 0.5); ax.set_ylim(0, 0.8)
    ax.set_xlabel("false-alarm rate on pre-plume frames (log)"); ax.set_ylabel("sensitivity on smoke-visible frames")
    ax.grid(True, color=GRID, linewidth=0.6, which="both"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6, loc="upper left"); ax.set_title("a. FIgLib operating curves (520 fires)", loc="left", color=INK)
    ax = axes[1]
    runs = sorted(ttd, key=lambda r: -ttd[r]["at_matched_far"]["0.05"]["camera_disjoint"]["sensitivity"])
    y = np.arange(len(runs))
    for k, (far, col) in enumerate((("0.01", ORD[1]), ("0.05", ORD[2]), ("0.1", ORD[4]))):
        v = [ttd[r]["at_matched_far"][far]["camera_disjoint"]["ttd_min_median"] for r in runs]
        ax.barh(y + (k - 1) * 0.26, v, height=0.24, color=col, label=f"FAR {float(far)*100:.0f} %", linewidth=0)
    ax.set_yticks(y); ax.set_yticklabels([figlib_label(r)[0].replace(" (", "\n").rstrip(")") for r in runs], fontsize=6); ax.invert_yaxis()
    ax.set_xlabel("median time-to-detection (min), held-out cameras"); ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, title="matched false-alarm rate", title_fontsize=6.5)
    ax.set_title("b. Detection delay at matched FAR", loc="left", color=INK)
    save(fig, "fig6_figlib", ["figlib_scores/<run>.json + coco_gt/figlib_all.json (ROC)", "figlib_ttd.json: runs.<run>.at_matched_far.<far>.camera_disjoint.ttd_min_median"])


def fig7_dedup_training():
    """Leakage at training time: clean-test vs official-test mAP50 for full / leaky-subset / de-duplicated training."""
    d = json.loads((HERE / "dedup_training_comparison.json").read_text(encoding="utf-8"))["runs"]
    groups = [("v8s", ["v8s_dfire_s3407", "v8s_dfire_s1337", "v8s_dfiresub_s3407", "v8s_dfiredd_s3407"]),
              ("y11s", ["y11s_dfire_s3407", "y11s_dfire_s1337", "y11s_dfiredd_s3407"])]
    lab = {"dfire_s3407": "full 15,499, seed 3407", "dfire_s1337": "full 15,499, seed 1337", "dfiresub_s3407": "leaky subset, 8,316", "dfiredd_s3407": "de-duplicated, 8,316"}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), gridspec_kw={"width_ratios": [4, 3]})
    for ax, (fam, runs) in zip(axes, groups):
        x = np.arange(len(runs)); w = 0.36
        clean = [d[r]["mAP50_d_fire_test_dedup"] for r in runs]; full = [d[r]["mAP50_d_fire_test"] for r in runs]
        ax.bar(x - w / 2, full, w * 0.95, color=ORD[1], label="official test (leaky)", linewidth=0)
        ax.bar(x + w / 2, clean, w * 0.95, color=ORD[3], label="clean test", linewidth=0)
        for i in range(len(runs)):
            ax.text(i - w / 2, full[i] + 0.01, f"{full[i]:.3f}", ha="center", fontsize=6, color=INK)
            ax.text(i + w / 2, clean[i] + 0.01, f"{clean[i]:.3f}", ha="center", fontsize=6, color=INK)
        ax.set_ylim(0.5, 0.85); ax.set_xticks(x); ax.set_xticklabels([lab[r.split("_", 1)[1]].replace(", ", "\n") for r in runs], fontsize=6.5)
        ax.set_title(f"{'a' if fam == 'v8s' else 'b'}. {FAM[fam[:-1]][0]}-s training set", loc="left", color=INK)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("mAP50 on D-Fire test")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2)
    save(fig, "fig7_dedup_training", ["dedup_training_comparison.json: runs.<run>.{mAP50_d_fire_test_dedup,mAP50_d_fire_test}"])


def main():
    summary = json.loads((HERE / "matrix_summary.json").read_text(encoding="utf-8"))
    fig1(summary)
    if (HERE / "threshold_decomposition.json").exists():
        fig2(json.loads((HERE / "threshold_decomposition.json").read_text(encoding="utf-8")))
    if (HERE / "dfire_leakage.json").exists():
        fig3(json.loads((HERE / "dfire_leakage.json").read_text(encoding="utf-8")))
    fig4(json.loads((HERE / "operating_points.json").read_text(encoding="utf-8")))
    fig5(HERE / "calibration_ci")
    if (HERE / "figlib_ttd.json").exists():
        fig6_figlib()
    if (HERE / "dedup_training_comparison.json").exists():
        fig7_dedup_training()
    (OUT / "figures_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
