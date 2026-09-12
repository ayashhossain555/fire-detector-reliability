"""Part C of label_free_revision.py (Holm-Bonferroni over the 44 M1c-vs-t_src cells) re-run at B = 10,000.

Referee point (2026-09-13): a two-sided bootstrap p from B = 1,000 resamples has resolution 2/B = 0.002, so it
cannot be compared with Holm's strictest level alpha/m = 0.05/44 = 0.00114, and "p < 0.001" cannot be stated.
This script calls label_free_revision.part_c UNCHANGED (imported read-only) with B_override = 10000 (seed 0 from
label_free_threshold_refine.json, same multinomial image-weight generator per target test; the point F1s are still
asserted equal to the refine artefact, only the B = 1000 CI / p reproduction asserts are skipped because B differs),
then puts every cell next to its B = 1000 row from label_free_revision.json (part C) and reports which cells
change Holm status. Two-sided p = min(1, 2 min(P(d <= 0), P(d >= 0))) has resolution 2/B = 0.0002 (one-sided 1e-4).
Writes analysis/label_free_holm_B10000.json (+ .md). Nothing else is written; label_free_revision.json is untouched.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import label_free_revision as lfr  # noqa: E402  read-only reuse: part_c (holm, ALPHA inside)

OUT_JSON = HERE / "label_free_holm_B10000.json"
OUT_MD = HERE / "label_free_holm_B10000.md"
REF_JSON = HERE / "label_free_revision.json"


def tag(s):
    return "gain" if s == "gain" else ("LOSS" if s == "loss" else "")


def write_md(out):
    C, R = out["C_B10000"], out["C_B1000_reference"]
    cmp_ = out["comparison"]
    changed = ", ".join(x["cell"] for x in cmp_["holm_status_changes"]) or "none"
    L = [f"# Holm-Bonferroni over {C['n_cells']} cells at B = {C['B']} (machine-written by label_free_holm_B10000.py)", "",
         f"Created {out['created']}; {out['seconds_total']} s CPU. Seed {C['seed']}, alpha {C['alpha']}, Holm strictest level alpha/m = {C['alpha'] / C['n_cells']:.5f}. "
         f"Two-sided p resolution 2/B = {2 / C['B']:g} (B = 10000) vs {2 / R['B']:g} (B = 1000). "
         f"Cells with two-sided p = 0: {C['n_cells_p_two_sided_equal_0']} (B = 10000) vs {R['n_cells_p_two_sided_equal_0']} (B = 1000).", "",
         "| group | n | Holm gain / loss B=10000 | B=1000 | Bonferroni gain / loss B=10000 | B=1000 | unadjusted gain / loss B=10000 | B=1000 | CI excl. 0 gain / loss B=10000 | B=1000 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for g in ("all", "yolo", "rtdetr"):
        c, r = C["counts"][g], R["counts"][g]
        L.append(f"| {g} | {c['n']} | {c['holm_gain']} / {c['holm_loss']} | {r['holm_gain']} / {r['holm_loss']} | "
                 f"{c['bonferroni_gain']} / {c['bonferroni_loss']} | {r['bonferroni_gain']} / {r['bonferroni_loss']} | "
                 f"{c['unadjusted_gain']} / {c['unadjusted_loss']} | {r['unadjusted_gain']} / {r['unadjusted_loss']} | "
                 f"{c['ci_excludes_0_gain']} / {c['ci_excludes_0_loss']} | {r['ci_excludes_0_gain']} / {r['ci_excludes_0_loss']} |")
    L += ["", f"Cells whose Holm status changes between B = 1000 and B = 10000: {cmp_['n_holm_status_changes']} ({changed}). "
              f"Max |change in point diff F1| = {cmp_['max_abs_point_diff_change']:.2e} (point estimates do not depend on B).", "",
          "| cell | family | diff F1 | CI95 B=10000 | CI95 B=1000 | p two-sided B=10000 | B=1000 | p Holm B=10000 | B=1000 | Holm B=10000 | B=1000 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in sorted(cmp_["cells"], key=lambda r: (r["p_two_sided_B10000"], r["p_two_sided_B1000"])):
        L.append(f"| {row['cell']} | {row['family']} | {row['point_diff_f1']:+.3f} | "
                 f"[{row['ci95_B10000'][0]:+.3f}, {row['ci95_B10000'][1]:+.3f}] | [{row['ci95_B1000'][0]:+.3f}, {row['ci95_B1000'][1]:+.3f}] | "
                 f"{row['p_two_sided_B10000']:.4f} | {row['p_two_sided_B1000']:.3f} | {row['p_holm_B10000']:.4f} | {row['p_holm_B1000']:.3f} | "
                 f"{tag(row['holm_B10000'])} | {tag(row['holm_B1000'])} |")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    return L


def status(r):
    if not r["significant_holm_0.05"]:
        return "ns"
    return "gain" if r["point_diff_f1"] > 0 else "loss"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args()
    t_start = time.time()
    refine = json.loads((HERE / "label_free_threshold_refine.json").read_text(encoding="utf-8"))
    ref_all = json.loads(REF_JSON.read_text(encoding="utf-8"))
    R = ref_all["C"]
    assert R["B"] == 1000 and R["seed"] == refine["bootstrap"]["seed"], (R["B"], R["seed"])
    C = lfr.part_c(refine, args.B)
    assert C["B"] == args.B and C["seed"] == R["seed"] and C["n_cells"] == R["n_cells"] == 44
    ref_rows = {r["cell"]: r for r in R["cells"]}
    cells, changes, max_dd = [], [], 0.0
    for r in C["cells"]:
        q = ref_rows[r["cell"]]
        max_dd = max(max_dd, abs(r["point_diff_f1"] - q["point_diff_f1"]))
        row = {"cell": r["cell"], "family": r["family"], "target": r["target"], "point_diff_f1": r["point_diff_f1"],
               "ci95_B10000": r["ci95"], "ci95_B1000": q["ci95"],
               "p_le_0_B10000": r["p_le_0"], "p_ge_0_B10000": r["p_ge_0"],
               "p_two_sided_B10000": r["p_two_sided_boot"], "p_two_sided_B1000": q["p_two_sided_boot"],
               "p_holm_B10000": r["p_holm_adjusted"], "p_holm_B1000": q["p_holm_adjusted"],
               "holm_B10000": status(r), "holm_B1000": status(q),
               "bonferroni_B10000": bool(r["significant_bonferroni_0.05"]), "bonferroni_B1000": bool(q["significant_bonferroni_0.05"])}
        cells.append(row)
        if row["holm_B10000"] != row["holm_B1000"]:
            changes.append({"cell": r["cell"], "from": row["holm_B1000"], "to": row["holm_B10000"],
                            "p_B1000": q["p_two_sided_boot"], "p_B10000": r["p_two_sided_boot"]})
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "protocol": __doc__,
           "inputs": {"refine": str(HERE / "label_free_threshold_refine.json"), "refine_created": refine["created"],
                      "reference_B1000": str(REF_JSON), "reference_created": ref_all["created"]},
           "C_B10000": C,
           "C_B1000_reference": {k: R[k] for k in ("B", "seed", "alpha", "n_cells", "resolution_note", "counts", "n_cells_p_two_sided_equal_0")},
           "comparison": {"holm_status_changes": changes, "n_holm_status_changes": len(changes), "max_abs_point_diff_change": max_dd,
                          "counts_delta_B10000_minus_B1000": {g: {k: C["counts"][g][k] - R["counts"][g][k] for k in C["counts"][g]} for g in C["counts"]},
                          "cells": cells}}
    out["seconds_total"] = round(time.time() - t_start, 1)
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    L = write_md(out)
    print("\n".join(L[:14]))
    print("artefact ->", OUT_JSON, f"({out['seconds_total']} s)")


if __name__ == "__main__":
    main()
