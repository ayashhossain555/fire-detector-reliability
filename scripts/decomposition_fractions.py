"""Decomposition fractions of the forward-cell LaECE_0 (referee item 3; 2026-09-13, CPU, no GPU).

Reads analysis/threshold_decomposition.json ONLY. For every cross-dataset cell the grid holds the
identity-map LaECE_0 of the target test under

    S/none    thr=source|map=none            (the forward cell: source LRP-optimal threshold, raw scores)
    T/none    thr=target|map=none            (target threshold alone)
    S/T-Platt thr=source|map=target:platt    (target-fitted Platt map alone, at the source threshold)
    T/T-Platt thr=target|map=target:platt    (both)

and reports, per cell, the fraction of the forward-cell LaECE_0 removed by
    (a) the target threshold alone      = (S/none - T/none)     / S/none
    (b) the target Platt map alone      = (S/none - S/T-Platt)  / S/none
    (c) both                            = (S/none - T/T-Platt)  / S/none
    interaction                         = (c) - (a) - (b)
(positive interaction: the two repairs remove more together than the sum of their separate effects;
negative: they overlap). The same fractions are given for the isotonic map and, for reference, for
the SOURCE-fitted maps (what a practitioner can apply without target labels). D_ECE fractions are
recorded in the JSON as a secondary key. Cells where a condition kept no detections (LaECE_0 = None)
are listed with the fraction = None. The forward-cell reproduction flag of threshold_decomposition.py
is carried through. Writes analysis/decomposition_fractions.json (+ .md).
"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "threshold_decomposition.json"
CORE_RUNS = ("v8s_dfire_s3407", "y11s_dfire_s3407", "rtdetrl_dfire_s3407",
             "v8s_pyrosdis_s3407", "y11s_pyrosdis_s3407", "rtdetrl_pyrosdis_s3407")
CONDS = {"S_none": "thr=source|map=none", "T_none": "thr=target|map=none",
         "S_Tplatt": "thr=source|map=target:platt", "T_Tplatt": "thr=target|map=target:platt",
         "S_Tiso": "thr=source|map=target:isotonic", "T_Tiso": "thr=target|map=target:isotonic",
         "S_Splatt": "thr=source|map=source:platt", "T_Splatt": "thr=target|map=source:platt",
         "S_Siso": "thr=source|map=source:isotonic", "T_Siso": "thr=target|map=source:isotonic"}


def frac(l0, l):
    if l0 is None or l is None or l0 <= 0:
        return None
    return float((l0 - l) / l0)


def fractions(v, map_tag, metric):
    """v: {cond_key: metric value}; map_tag encodes map origin (T/S) and family (platt/iso)."""
    l0 = v["S_none"]
    a = frac(l0, v["T_none"])
    b = frac(l0, v[f"S_{map_tag}"])
    c = frac(l0, v[f"T_{map_tag}"])
    inter = None if None in (a, b, c) else float(c - a - b)
    return {"forward_" + metric: l0, "threshold_alone": a, "map_alone": b, "both": c, "interaction": inter,
            "absolute_removed": {"threshold_alone": None if a is None else float(l0 - v["T_none"]),
                                 "map_alone": None if b is None else float(l0 - v[f"S_{map_tag}"]),
                                 "both": None if c is None else float(l0 - v[f"T_{map_tag}"])}}


def is_core(c):
    if c["run"] not in CORE_RUNS or c["test"].endswith("_dedup"):
        return False
    # pyro-trained (smoke-only) models: the smoke-only evaluation is the core one; the fire+smoke GT variant is a duplicate view
    if "pyrosdis" in c["run"] and c["test"] in ("d_fire_test", "thesis_test"):
        return False
    return True


def summarise(sel, key_map, metric):
    out = {}
    for comp in ("threshold_alone", "map_alone", "both", "interaction"):
        x = np.array([s["fractions"][metric][key_map][comp] for s in sel
                      if s["fractions"][metric][key_map][comp] is not None], dtype=float)
        out[comp] = {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None, "median": float(np.median(x)) if len(x) else None,
                     "min": float(x.min()) if len(x) else None, "max": float(x.max()) if len(x) else None}
    ab = [(s["fractions"][metric][key_map]["threshold_alone"], s["fractions"][metric][key_map]["map_alone"]) for s in sel]
    ab = [(a, b) for a, b in ab if a is not None and b is not None]
    out["n_threshold_removes_more_than_map"] = int(sum(1 for a, b in ab if a > b))
    out["n_map_removes_more_than_threshold"] = int(sum(1 for a, b in ab if b > a))
    out["n_interaction_negative"] = int(sum(1 for s in sel if (s["fractions"][metric][key_map]["interaction"] or 0) < 0))
    return out


def main():
    t0 = time.time()
    td = json.loads(SRC.read_text(encoding="utf-8"))
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "source_artefact": str(SRC), "source_grid": td["grid"],
           "definition": __doc__, "core_runs": list(CORE_RUNS), "cells": {}}
    for cell, c in sorted(td["cells"].items()):
        r = c["results"]
        vals = {m: {k: r[cond].get(m) for k, cond in CONDS.items()} for m in ("LaECE_0", "D_ECE")}
        fr = {}
        for m in ("LaECE_0", "D_ECE"):
            fr[m] = {"target_platt": fractions(vals[m], "Tplatt", m), "target_isotonic": fractions(vals[m], "Tiso", m),
                     "source_platt": fractions(vals[m], "Splatt", m), "source_isotonic": fractions(vals[m], "Siso", m)}
        out["cells"][cell] = {"run": c["run"], "source_calval": c["source_calval"], "target_calval": c["target_calval"], "test": c["test"],
                              "family": "rtdetr" if c["run"].startswith("rtdetr") else "yolo",
                              "core": is_core(c),
                              "forward_identity_reproduced": r["check_reproduces_forward_identity"]["ok"],
                              "n_dets_kept": {"S": r[CONDS["S_none"]]["n_dets_kept"], "T": r[CONDS["T_none"]]["n_dets_kept"]},
                              "values": vals, "fractions": fr}
    cells = out["cells"]
    groups = {"all": list(cells.values()), "core": [c for c in cells.values() if c["core"]],
              "yolo": [c for c in cells.values() if c["family"] == "yolo"], "rtdetr": [c for c in cells.values() if c["family"] == "rtdetr"]}
    out["summary"] = {g: {m: {km: summarise(sel, km, m) for km in ("target_platt", "target_isotonic", "source_platt", "source_isotonic")}
                          for m in ("LaECE_0", "D_ECE")} for g, sel in groups.items() if sel}
    out["summary"]["n_cells"] = len(cells)
    out["summary"]["n_core"] = len(groups["core"])
    out["summary"]["n_forward_identity_not_reproduced"] = int(sum(1 for c in cells.values() if not c["forward_identity_reproduced"]))
    out["seconds"] = round(time.time() - t0, 2)
    (HERE / "decomposition_fractions.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    def f3(x):
        return "" if x is None else f"{x:.3f}"

    L = ["# Decomposition fractions of the forward-cell LaECE_0 (machine-written by decomposition_fractions.py)", "",
         f"Source: threshold_decomposition.json ({len(cells)} cross cells; forward identity reproduced in "
         f"{len(cells) - out['summary']['n_forward_identity_not_reproduced']}/{len(cells)}). "
         "Fraction removed = (S/none - X) / S/none; interaction = both - threshold_alone - map_alone.", ""]
    for g in ("core", "all", "yolo", "rtdetr"):
        if g not in out["summary"]:
            continue
        L += [f"## Summary - {g} (n = {len(groups[g])}), LaECE_0", "",
              "| map | component | mean | median | min | max | n thr>map | n map>thr | n interaction<0 |", "|---|---|---|---|---|---|---|---|---|"]
        for km in ("target_platt", "target_isotonic", "source_platt"):
            s = out["summary"][g]["LaECE_0"][km]
            for comp in ("threshold_alone", "map_alone", "both", "interaction"):
                e = s[comp]
                first = comp == "threshold_alone"
                L.append(f"| {km} | {comp} | {f3(e['mean'])} | {f3(e['median'])} | {f3(e['min'])} | {f3(e['max'])} | "
                         f"{s['n_threshold_removes_more_than_map'] if first else ''} | "
                         f"{s['n_map_removes_more_than_threshold'] if first else ''} | "
                         f"{s['n_interaction_negative'] if first else ''} |")
        L.append("")
    L += ["## Per cell (LaECE_0; target-fitted Platt unless stated)", "",
          "| cell | core | S/none | T/none | S/T-Platt | T/T-Platt | (a) thr alone | (b) map alone | (c) both | interaction | (b') S-Platt alone | (c') thr + S-Platt | iso (a)/(b)/(c) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell, c in sorted(cells.items(), key=lambda kv: (not kv[1]["core"], kv[0])):
        v = c["values"]["LaECE_0"]
        fp = c["fractions"]["LaECE_0"]["target_platt"]
        fs = c["fractions"]["LaECE_0"]["source_platt"]
        fi = c["fractions"]["LaECE_0"]["target_isotonic"]
        L.append(f"| {cell} | {'yes' if c['core'] else ''} | {f3(v['S_none'])} | {f3(v['T_none'])} | {f3(v['S_Tplatt'])} | {f3(v['T_Tplatt'])} | "
                 f"{f3(fp['threshold_alone'])} | {f3(fp['map_alone'])} | {f3(fp['both'])} | {f3(fp['interaction'])} | "
                 f"{f3(fs['map_alone'])} | {f3(fs['both'])} | {f3(fi['threshold_alone'])}/{f3(fi['map_alone'])}/{f3(fi['both'])} |")
    (HERE / "decomposition_fractions.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))
    print("artefact ->", HERE / "decomposition_fractions.json", "and .md", f"({out['seconds']} s)")


if __name__ == "__main__":
    main()
