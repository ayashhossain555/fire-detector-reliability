"""Side-by-side of the B = 10000 paired post-hoc-map bootstrap (calibration_ci_B10000/<cell>.json, written by
bootstrap_ci_B10000.py) against the B = 1000 artefacts (calibration_ci/<cell>.json, bootstrap_ci.py), per cell and
per calibrator, for LaECE_0 and D_ECE: point difference vs the identity map (B-independent, asserted equal to 1e-9),
percentile 95 % CI, two-sided bootstrap p (resolution 2/B: 0.0002 at B = 10000, 0.002 at B = 1000), and whether the
'CI excludes 0 / p < 0.05' significance call changes. Reads only; writes calibration_ci_B10000/_paired_comparison.json
(+ .md). calibration_ci/ is never written.
"""
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW, REF = HERE / "calibration_ci_B10000", HERE / "calibration_ci"
CALS = ("temperature_scaling", "platt_scaling", "isotonic_regression")
METRICS = ("LaECE_0", "D_ECE")


def sig(d):
    return "harms" if d["ci95"][0] > 0 else ("helps" if d["ci95"][1] < 0 else "ns")


def main():
    cells, changes = {}, []
    for p in sorted(NEW.glob("*.json")):
        if p.name.startswith("_"):
            continue
        n = json.loads(p.read_text(encoding="utf-8"))
        r = json.loads((REF / p.name).read_text(encoding="utf-8"))
        assert n["B"] == 10000 and r["B"] == 1000 and n["seed"] == r["seed"] == 3407, (p.name, n["B"], r["B"])
        row = {"B_new": n["B"], "B_ref": r["B"], "seed": n["seed"], "wall_s_new": n["wall_s"], "wall_s_ref": r["wall_s"], "maps": {}}
        for cal in CALS:
            row["maps"][cal] = {}
            for m in METRICS:
                a, b = n["paired_vs_identity"][cal][m], r["paired_vs_identity"][cal][m]
                assert abs(a["point_diff"] - b["point_diff"]) < 1e-9, (p.name, cal, m, a["point_diff"], b["point_diff"])
                for k in ("identity", cal):
                    assert abs(n["results"][k][m]["point"] - r["results"][k][m]["point"]) < 1e-9, (p.name, k, m)
                e = {"point_diff": a["point_diff"], "ci95_B10000": a["ci95"], "ci95_B1000": b["ci95"],
                     "p_two_sided_B10000": a["p_two_sided_boot"], "p_two_sided_B1000": b["p_two_sided_boot"],
                     "call_B10000": sig(a), "call_B1000": sig(b),
                     "p_lt_0.001_at_B10000": bool(a["p_two_sided_boot"] < 0.001)}
                row["maps"][cal][m] = e
                if e["call_B10000"] != e["call_B1000"]:
                    changes.append({"cell": p.stem, "map": cal, "metric": m, "from": e["call_B1000"], "to": e["call_B10000"],
                                    "p_B1000": b["p_two_sided_boot"], "p_B10000": a["p_two_sided_boot"]})
        cells[p.stem] = row
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "protocol": __doc__, "n_cells": len(cells),
           "p_resolution": {"B10000": 2 / 10000, "B1000": 2 / 1000},
           "n_significance_call_changes": len(changes), "significance_call_changes": changes, "cells": cells}
    (NEW / "_paired_comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    L = [f"# Paired post-hoc map vs identity: B = 10000 vs B = 1000 ({len(cells)} cells; machine-written by bootstrap_ci_B10000_compare.py, {out['created']})", "",
         "LaECE_0 in percentage points (x100). Two-sided bootstrap p resolution 0.0002 (B = 10000) vs 0.002 (B = 1000). "
         f"Significance-call changes (CI excludes 0): {len(changes)}.", "",
         "| cell | map | diff LaECE_0 (pp) | CI95 B=10000 | CI95 B=1000 | p B=10000 | p B=1000 | call B=10000 | B=1000 | diff D_ECE (pp) | p D_ECE B=10000 | B=1000 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, row in cells.items():
        for cal in CALS:
            e, d = row["maps"][cal]["LaECE_0"], row["maps"][cal]["D_ECE"]
            L.append(f"| {c} | {cal.split('_')[0]} | {e['point_diff'] * 100:+.1f} | [{e['ci95_B10000'][0] * 100:+.1f}, {e['ci95_B10000'][1] * 100:+.1f}] | "
                     f"[{e['ci95_B1000'][0] * 100:+.1f}, {e['ci95_B1000'][1] * 100:+.1f}] | {e['p_two_sided_B10000']:.4f} | {e['p_two_sided_B1000']:.3f} | "
                     f"{e['call_B10000']} | {e['call_B1000']} | {d['point_diff'] * 100:+.1f} | {d['p_two_sided_B10000']:.4f} | {d['p_two_sided_B1000']:.3f} |")
    if changes:
        L += ["", "Changes: " + "; ".join(f"{x['cell']} {x['map']} {x['metric']} {x['from']} -> {x['to']} (p {x['p_B1000']:.3f} -> {x['p_B10000']:.4f})" for x in changes)]
    (NEW / "_paired_comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("artefact ->", NEW / "_paired_comparison.json")


if __name__ == "__main__":
    main()
