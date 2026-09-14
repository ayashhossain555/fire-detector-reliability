"""Paper 05 — scale (n/s/m) and seed (3407 vs 1337) views of the calibration
matrix, read from matrix_summary.json (which is rebuilt by summarise_matrix.py
after every run). Writes scale_seed_summary.json + .md.

Tables
  A. scale  : per source × family × size (seed 3407, plain variant): val
              mAP50, in-domain LaECE_0 (identity, ×100, with CI when present),
              D-ECE, LRP-FN, and the cross-target LaECE_0 identity for every
              target tag, plus image-level sensitivity / FAR at t_src.
  B. seeds  : for every run with both seeds: the same columns per seed and
              the absolute seed difference; summary of |Δ| per column.
  C. spread : per (source, target_tag) group over all runs: min / median /
              max of LaECE_0 identity and of val mAP50 — the "same mAP,
              wide calibration spread" statement with the full model set.
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = json.loads((HERE / "matrix_summary.json").read_text(encoding="utf-8"))
runs, cells = M["runs"], M["cells"]

SIZE_ORDER = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}


def parse_run(name):
    # 2026-09-14: rtdetrx (RT-DETR-x) added; parsed as family "rtdetr", size "x" (rtdetrl unchanged)
    m = re.match(r"(v8|y11|rtdetrl|rtdetrx)([nsml])?_(dfire|pyrosdis|dfiredd|dfiresub)_s(\d+)(_p60)?$", name)
    if not m:
        return None
    fam, size, src, seed, p60 = m.groups()
    if fam in ("rtdetrl", "rtdetrx"):
        fam, size = "rtdetr", fam[-1]
    variant = {"dfiredd": "dedup", "dfiresub": "subset"}.get(src, "") + ("p60" if p60 else "")
    src = "dfire" if src.startswith("dfire") else src
    return {"family": fam, "size": size, "source": src, "seed": int(seed), "variant": variant}


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def cell_row(c):
    met = c.get("metrics", {})
    ident = met.get("identity", {})
    row = {"LaECE_id": ident.get("LaECE_0"), "D_ECE_id": ident.get("D_ECE"), "lrp_fn_id": ident.get("lrp_fn"),
           "LaECE_platt": g(met, "platt_scaling", "LaECE_0"), "LaECE_temp": g(met, "temperature_scaling", "LaECE_0"),
           "LaECE_iso": g(met, "isotonic", "LaECE_0", default=g(met, "isotonic_regression", "LaECE_0"))}
    row["LaECE_id_ci"] = g(c, "ci95", "identity", "LaECE_0")
    op = c.get("operating_point") or {}
    img = op.get("image_level_at_t_src") or {}
    row["t_src"] = op.get("t_src"); row["target_f1_at_t_src"] = op.get("target_f1_at_t_src"); row["target_f1_opt"] = op.get("target_f1_opt")
    row["sens_t_src"] = img.get("sensitivity"); row["far_t_src"] = img.get("false_alarm_rate")
    row["platt_paired"] = g(c, "paired_vs_identity", "platt_scaling", "LaECE_0")
    return row


def main():
    by_run = {}
    for name, c in cells.items():
        if c.get("kind") not in ("in_domain", "cross") or "__repair" in name:
            continue  # target-fitted repair cells share run/target_tag; keep forward cells only
        r = c["run"]
        by_run.setdefault(r, {})[c["target_tag"]] = cell_row(c)
    # first-cell field dump for provenance
    sample = next(iter(cells.values()))
    out = {"generated_from": "matrix_summary.json", "matrix_generated": M.get("generated"), "n_runs": M["n_runs"], "n_cells": M["n_cells"],
           "cell_fields_seen": sorted(sample.keys()), "runs": {}}
    for r in runs:
        p = parse_run(r)
        if not p:
            continue
        rr = runs[r]
        out["runs"][r] = {**p, "best_val_mAP50": rr.get("best_val_mAP50"), "best_epoch": rr.get("best_epoch"), "epochs_run": rr.get("epochs_run"),
                          "wall_h": (rr.get("wall_s_total") or 0) / 3600.0, "cells": by_run.get(r, {})}
    # A. scale table (seed 3407, plain variant)
    A = []
    for r, v in out["runs"].items():
        if v["seed"] != 3407 or v["variant"]:
            continue
        src = v["source"]
        in_tag = "d_fire_test" if src == "dfire" else "pyro_sdis_caltest"
        cross = {"pyro_sdis_caltest": "→Pyro-SDIS", "d_fire_test_smokeonly": "→D-Fire smoke", "d_fire_test": "→D-Fire",
                 "thesis_test": "→thesis", "thesis_test_smokeonly": "→thesis smoke", "figlib_bb": "→FIgLib boxes"}
        row = {"run": r, "source": src, "family": v["family"], "size": v["size"], "val_mAP50": v["best_val_mAP50"],
               "in_domain": v["cells"].get(in_tag, {})}
        row["cross"] = {lab: v["cells"][t] for t, lab in cross.items() if t in v["cells"] and t != in_tag}
        A.append(row)
    A.sort(key=lambda x: (x["source"], x["family"], SIZE_ORDER.get(x["size"], 9)))
    out["A_scale"] = A
    # B. seeds
    B = {}
    for r, v in out["runs"].items():
        if v["variant"]:
            continue
        key = f"{v['family']}{'' if (v['family']=='rtdetr' and v['size']=='l') else v['size']}_{v['source']}"  # rtdetr_<src> = RT-DETR-l (unchanged); rtdetrx_<src> = RT-DETR-x
        B.setdefault(key, {})[v["seed"]] = {"run": r, "val_mAP50": v["best_val_mAP50"], "cells": v["cells"]}
    seeds = {}
    diffs = {"val_mAP50": [], "LaECE_id_in": [], "LaECE_id_cross_main": []}
    for key, per_seed in B.items():
        if len(per_seed) < 2:
            continue
        a, b = per_seed[3407], per_seed[1337]
        src = key.split("_")[-1]
        in_tag = "d_fire_test" if src == "dfire" else "pyro_sdis_caltest"
        main_tag = "pyro_sdis_caltest" if src == "dfire" else "d_fire_test_smokeonly"
        rec = {"seed3407": a["run"], "seed1337": b["run"],
               "val_mAP50": [a["val_mAP50"], b["val_mAP50"]],
               "LaECE_id_in": [g(a["cells"], in_tag, "LaECE_id"), g(b["cells"], in_tag, "LaECE_id")],
               "LaECE_id_cross_main": [g(a["cells"], main_tag, "LaECE_id"), g(b["cells"], main_tag, "LaECE_id")],
               "sens_cross_main": [g(a["cells"], main_tag, "sens_t_src"), g(b["cells"], main_tag, "sens_t_src")]}
        for k in diffs:
            x, y = rec[k]
            if x is not None and y is not None:
                rec[f"abs_diff_{k}"] = abs(x - y); diffs[k].append(abs(x - y))
        seeds[key] = rec
    out["B_seeds"] = seeds
    out["B_seed_abs_diff_summary"] = {k: {"n": len(v), "median": st.median(v) if v else None, "max": max(v) if v else None} for k, v in diffs.items()}
    # C. spread per (source, target_tag) over ALL plain-variant runs incl. sizes and seeds
    C = {}
    for r, v in out["runs"].items():
        if v["variant"]:
            continue
        for tag, row in v["cells"].items():
            if row.get("LaECE_id") is None:
                continue
            C.setdefault(f"{v['source']}__to__{tag}", []).append((r, v["best_val_mAP50"], row["LaECE_id"]))
    out["C_spread"] = {k: {"n_runs": len(v), "runs": [x[0] for x in v],
                           "val_mAP50_min_med_max": [min(x[1] for x in v), st.median(x[1] for x in v), max(x[1] for x in v)],
                           "LaECE_id_min_med_max": [min(x[2] for x in v), st.median(x[2] for x in v), max(x[2] for x in v)]}
                       for k, v in C.items()}
    (HERE / "scale_seed_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    # markdown
    L = ["# Scale / seed summary (from matrix_summary.json; LaECE ×100)", "", "## A. Scale (seed 3407)", "",
         "| run | val mAP50 | in-domain LaECE id | D-ECE | LRP-FN | cross LaECE id (per target) | sens / FAR @t_src (main cross) |", "|---|---|---|---|---|---|---|"]
    f = lambda x, m=100: "—" if x is None else (f"{x*m:.1f}" if m == 100 else f"{x:.3f}")
    for a in A:
        i = a["in_domain"]
        cross = "; ".join(f"{lab} {f(c['LaECE_id'])}" for lab, c in a["cross"].items())
        main = a["cross"].get("→Pyro-SDIS") or a["cross"].get("→D-Fire smoke") or {}
        ci = i.get('LaECE_id_ci'); cis = f" [{ci[0]*100:.1f}, {ci[1]*100:.1f}]" if ci else ""
        L.append(f"| {a['run']} | {f(a['val_mAP50'],1)} | {f(i.get('LaECE_id'))}{cis} | {f(i.get('D_ECE_id'))} | {f(i.get('lrp_fn_id'))} | {cross} | {f(main.get('sens_t_src'),1)} / {f(main.get('far_t_src'),1)} |")
    L += ["", "## B. Seeds (3407 vs 1337)", "", "| model | val mAP50 | in-domain LaECE id | cross-main LaECE id | cross-main sens@t_src |", "|---|---|---|---|---|"]
    for k, rec in seeds.items():
        p = lambda xs, m=100: " / ".join(f(x, m) for x in xs)
        L.append(f"| {k} | {p(rec['val_mAP50'],1)} | {p(rec['LaECE_id_in'])} | {p(rec['LaECE_id_cross_main'])} | {p(rec['sens_cross_main'],1)} |")
    L += ["", f"|Δ| summary: {json.dumps(out['B_seed_abs_diff_summary'])}", "", "## C. Spread per (source → target) over all plain runs", "",
          "| group | n | val mAP50 min/med/max | LaECE id min/med/max |", "|---|---|---|---|"]
    for k, v in sorted(out["C_spread"].items()):
        L.append(f"| {k} | {v['n_runs']} | {' / '.join(f'{x:.3f}' for x in v['val_mAP50_min_med_max'])} | {' / '.join(f'{x*100:.1f}' for x in v['LaECE_id_min_med_max'])} |")
    (HERE / "scale_seed_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
