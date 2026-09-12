"""Aggregate every artefact into analysis/matrix_summary.json (+ a Markdown
table analysis/matrix_summary.md) — the single place the draft reads from.

Sources (all machine-written):
  runs_meta/*.json            training runs (mAP, checkpoint SHA)
  calibration/*.json          LaECE_0 / D-ECE / LRP per cell x calibrator
  calibration_ci/*.json       image-bootstrap 95% CIs + paired diffs vs identity
  operating_points.json       shipped-threshold deployment metrics
Cell naming: <run>__to__<target>[__repair]. Derived fields per cell:
  family (v8/y11/rtdetr), size (n/s/m/l), source dataset, seed, target
  dataset, kind (in_domain / cross / repair), smoke_only flag.
"""
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAL, CI, META = HERE / "calibration", HERE / "calibration_ci", HERE / "runs_meta"
CALIBRATORS = ["identity", "temperature_scaling", "platt_scaling", "isotonic_regression"]
TARGET_DS = {"d_fire": "dfire", "pyro_sdis": "pyrosdis", "thesis": "thesis", "figlib": "figlib"}


def parse_run(run):
    m = re.match(r"^(v8|y11|rtdetr)([nsmlx])_(dfire|pyrosdis|dfiredd|dfiresub)_s(\d+)(_[a-z0-9]+)?$", run)
    return {"family": m.group(1), "size": m.group(2), "source": m.group(3), "seed": int(m.group(4)),
            "variant": (m.group(5) or "").lstrip("_")} if m else {}


def parse_cell(name):
    run, rest = name.split("__to__")
    repair = rest.endswith("__repair")
    tgt = rest[:-8] if repair else rest
    ds = next((v for k, v in TARGET_DS.items() if tgt.startswith(k)), tgt)
    info = parse_run(run)
    kind = "repair" if repair else ("in_domain" if ds == info.get("source") else "cross")
    return {"run": run, **info, "target_tag": tgt, "target": ds, "kind": kind,
            "smoke_only": tgt.endswith("smokeonly")}


def main():
    runs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(META.glob("*.json"))}
    ops = json.loads((HERE / "operating_points.json").read_text(encoding="utf-8"))["cells"] \
        if (HERE / "operating_points.json").exists() else {}
    cells = {}
    for p in sorted(CAL.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        c = parse_cell(d["cell"])
        c["metrics"] = {cal: {k: d["results"][cal][k] for k in ("LaECE_0", "LaACE_0", "D_ECE", "lrp", "lrp_loc", "lrp_fp", "lrp_fn")}
                        for cal in CALIBRATORS if cal in d["results"]}
        cip = CI / p.name
        if cip.exists():
            ci = json.loads(cip.read_text(encoding="utf-8"))
            c["ci95"] = {cal: {m: ci["results"][cal][m]["ci95"] for m in ("LaECE_0", "D_ECE")} for cal in CALIBRATORS if cal in ci["results"]}
            c["paired_vs_identity"] = ci["paired_vs_identity"]
            c["ci_B"] = ci["B"]
        if d["cell"] in ops:
            o = ops[d["cell"]]
            c["operating_point"] = {"t_src": o["source"]["t_f1opt"], "src_f1": o["source"]["f1_at_t"],
                                    "target_f1_at_t_src": (o["target"]["box_at_t_src"] or {}).get("f1"),
                                    "target_f1_opt": o["target"]["f1_at_target_opt"], "threshold_drift": o["target"]["threshold_drift"],
                                    "image_level_at_t_src": o["target"]["image_level_at_t_src"],
                                    "t_for_95pct_sensitivity": o["target"]["t_for_95pct_sensitivity"],
                                    "image_level_at_t95": o["target"]["image_level_at_t95"]}
        cells[d["cell"]] = c
    # roll-ups: per (family,size,source,target,kind) across seeds
    groups = {}
    for name, c in cells.items():
        if not c.get("family"):
            continue
        key = f"{c['family']}{c['size']}|{c['source']}->{c['target_tag']}|{c['kind']}"
        g = groups.setdefault(key, {"cells": [], "seeds": []})
        g["cells"].append(name); g["seeds"].append(c["seed"])
    for key, g in groups.items():
        for cal in CALIBRATORS:
            vals = [cells[n]["metrics"][cal]["LaECE_0"] for n in g["cells"] if cal in cells[n]["metrics"]]
            if vals:
                g[f"LaECE_0_{cal}"] = {"mean": float(np.mean(vals)), "min": float(min(vals)), "max": float(max(vals)), "n_seeds": len(vals)}
    out = {"generated": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"), "n_runs": len(runs), "n_cells": len(cells),
           "runs": runs, "cells": cells, "groups_across_seeds": groups}
    (HERE / "matrix_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    # markdown table
    lines = ["| cell | kind | mAP50 (src val) | LaECE_0 id | 95% CI | temp | Platt | iso | D-ECE id | LRP-FN id | img sens@t_src | FAR@t_src |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in sorted(cells.items(), key=lambda x: (x[1].get("kind", ""), x[0])):
        m = c["metrics"]; ci = c.get("ci95", {}).get("identity", {}).get("LaECE_0")
        op = c.get("operating_point", {}); il = (op.get("image_level_at_t_src") or {})
        f = lambda v: f"{v*100:.1f}" if v is not None else "—"
        lines.append(f"| {name} | {c.get('kind','')} | {runs.get(c['run'],{}).get('best_val_mAP50','—')} | {f(m['identity']['LaECE_0'])} | "
                     f"{'[' + f(ci[0]) + ', ' + f(ci[1]) + ']' if ci else '—'} | {f(m.get('temperature_scaling',{}).get('LaECE_0'))} | "
                     f"{f(m.get('platt_scaling',{}).get('LaECE_0'))} | {f(m.get('isotonic_regression',{}).get('LaECE_0'))} | "
                     f"{f(m['identity']['D_ECE'])} | {f(m['identity']['lrp_fn'])} | {f(il.get('sensitivity'))} | {f(il.get('false_alarm_rate'))} |")
    (HERE / "matrix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"runs {len(runs)}, cells {len(cells)}, groups {len(groups)} -> matrix_summary.json / .md")


if __name__ == "__main__":
    main()
