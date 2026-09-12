"""Label-free threshold transfer, broken down by transfer direction (referee item).

Reads ONLY the stored per-cell numbers of
  analysis/label_free_threshold_refine.json  (methods.M0_t_src / D25_default / M1c_kept_rate /
                                              ORC_target_calval .f1, and their image-bootstrap CIs)
  analysis/label_free_threshold_pilot.json   (methods.M1a_quantile_pooled.box.f1 - the quantile-
                                              pooled rule, which the refine dropped)
and re-summarises them for the four transfer directions

  D-Fire -> Pyro-SDIS            (13 cells: 10 dfire + 2 dfiredd + 1 dfiresub trained runs)
  D-Fire -> thesis               (13 cells)
  Pyro-SDIS -> D-Fire (smoke)    ( 9 cells)
  Pyro-SDIS -> thesis (smoke)    ( 9 cells)

for the subsets "all" and "oracle F1 >= 0.10" (cells whose target-calval-oracle threshold reaches
a target-test box F1 of at least 0.10; below that floor the F1 gap is a small number divided by
another small number and "gap closed" is not meaningful).

Per direction x subset x rule (D25 default 0.25, M1c kept-rate, M1a quantile-pooled):
  n cells, mean F1 at t_src / 0.25 / kept-rate / quantile-pooled / oracle,
  mean gap = mean(F1_oracle - F1_t_src), mean regret = mean(F1_oracle - F1_rule),
  gap closed = 1 - mean regret / mean gap  (ratio of means, NOT mean of per-cell ratios),
  cell-level bootstrap (resample cells with replacement, seed 0, B = 10,000, percentile 2.5-97.5)
  of the mean regret and of gap closed, beats / worse / ties vs t_src (|dF1| <= 1e-12 = tie, as in
  the refine's n_beats_t_src / n_worse_than_t_src).
Plus the paired per-cell difference  F1(kept-rate) - F1(quantile-pooled)  (mean, cell bootstrap CI,
wins / losses / ties), overall and per direction.

The stored per-cell image-level CIs (refine) are carried through for reference; no image-level
replicates are re-drawn - the refine does not store them and the cell-level question (is the
0.009 vs 0.014 mean regret difference distinguishable across cells?) is answered at cell level.

Writes analysis/label_free_by_direction.json and .md. CPU only, seconds.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REFINE = HERE / "label_free_threshold_refine.json"
PILOT = HERE / "label_free_threshold_pilot.json"
OUT_JSON = HERE / "label_free_by_direction.json"
OUT_MD = HERE / "label_free_by_direction.md"
SEED = 0
B = 10_000
FLOOR = 0.10
TIE = 1e-12
RULES = {"D25_default": "default 0.25", "M1c_kept_rate": "kept-rate (refine M1c)", "M1a_quantile_pooled": "quantile-pooled (pilot M1a)"}

DIRECTIONS = {  # (train_set family, target) -> label
    ("dfire", "pyro_sdis"): "D-Fire->Pyro-SDIS", ("dfire", "thesis"): "D-Fire->thesis",
    ("pyrosdis", "d_fire"): "Pyro-SDIS->D-Fire (smoke)", ("pyrosdis", "thesis"): "Pyro-SDIS->thesis (smoke)"}
DIR_ORDER = ["D-Fire->Pyro-SDIS", "D-Fire->thesis", "Pyro-SDIS->D-Fire (smoke)", "Pyro-SDIS->thesis (smoke)"]


def direction_of(cell):
    ts = cell["train_set"]
    fam = "dfire" if ts.startswith("dfire") else ("pyrosdis" if ts.startswith("pyrosdis") else ts)
    return DIRECTIONS[(fam, cell["target"])]


def load_cells():
    r = json.loads(REFINE.read_text(encoding="utf-8"))
    p = json.loads(PILOT.read_text(encoding="utf-8"))
    cells = {}
    for name, c in r["cells"].items():
        m = c["methods"]
        pm = p["cells"][name]["methods"]
        # cross-check: pilot and refine agree on the shared rules (same detections, same evaluation)
        for k in ("M0_t_src", "D25_default", "M1c_kept_rate", "ORC_target_calval"):
            assert abs(pm[k]["box"]["f1"] - m[k]["f1"]) < 1e-9, f"{name} {k}: pilot {pm[k]['box']['f1']} vs refine {m[k]['f1']}"
        f1 = {"M0_t_src": m["M0_t_src"]["f1"], "D25_default": m["D25_default"]["f1"], "M1c_kept_rate": m["M1c_kept_rate"]["f1"],
              "ORC_target_calval": m["ORC_target_calval"]["f1"], "M1a_quantile_pooled": pm["M1a_quantile_pooled"]["box"]["f1"]}
        thr = {"M0_t_src": m["M0_t_src"]["threshold"], "D25_default": 0.25, "M1c_kept_rate": m["M1c_kept_rate"]["threshold"],
               "ORC_target_calval": m["ORC_target_calval"]["threshold"], "M1a_quantile_pooled": pm["M1a_quantile_pooled"]["threshold"]}
        gap = f1["ORC_target_calval"] - f1["M0_t_src"]
        assert abs(gap - m["M0_t_src"]["regret_vs_oracle"]) < 1e-9
        cells[name] = {"direction": direction_of(c), "run": c["run"], "family": c["family"], "train_set": c["train_set"], "target": c["target"],
                       "target_test": c["target_test"], "n_target_test_images": c["n"]["target_test_images"], "n_target_test_gt": c["n"]["target_test_gt"],
                       "f1": f1, "threshold": thr, "gap_oracle_minus_t_src": gap,
                       "regret": {k: f1["ORC_target_calval"] - f1[k] for k in RULES},
                       "oracle_f1_ge_floor": bool(f1["ORC_target_calval"] >= FLOOR),
                       "stored_image_ci95_f1": {k: m[k]["ci95"]["f1"] for k in ("M0_t_src", "D25_default", "M1c_kept_rate", "ORC_target_calval")},
                       "stored_image_ci95_regret_vs_oracle": {k: m[k]["regret_vs_oracle_ci95"] for k in ("M0_t_src", "D25_default", "M1c_kept_rate")},
                       "stored_gap_ci95": m["ORC_target_calval"]["gap_ci95"],
                       "source_keys": {"refine": f"cells.{name}.methods.<rule>.f1", "pilot": f"cells.{name}.methods.M1a_quantile_pooled.box.f1"}}
    return cells, r, p


def ci(x):
    x = np.asarray(x, dtype=float)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def summarise(sel, cells):
    """sel: list of cell names. Paired cell bootstrap: one index matrix shared by every rule."""
    n = len(sel)
    out = {"n_cells": n, "cells": sel}
    if n == 0:
        return out
    F = {k: np.array([cells[c]["f1"][k] for c in sel]) for k in ("M0_t_src", "D25_default", "M1c_kept_rate", "M1a_quantile_pooled", "ORC_target_calval")}
    gap = F["ORC_target_calval"] - F["M0_t_src"]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))
    gap_b = gap[idx].mean(axis=1)
    out["mean_f1"] = {k: float(v.mean()) for k, v in F.items()}
    out["mean_gap_oracle_minus_t_src"] = float(gap.mean())
    out["mean_gap_ci95_cell_bootstrap"] = ci(gap_b)
    out["bootstrap"] = {"unit": "cell", "B": B, "seed": SEED, "ci": "percentile 2.5-97.5", "paired_across_rules": True}
    out["rules"] = {}
    for k in RULES:
        reg = F["ORC_target_calval"] - F[k]
        reg_b = reg[idx].mean(axis=1)
        gc = 1.0 - reg.mean() / gap.mean() if gap.mean() > 1e-9 else None
        with np.errstate(divide="ignore", invalid="ignore"):
            gc_b = 1.0 - reg_b / gap_b
        gc_ok = np.isfinite(gc_b)
        d = F[k] - F["M0_t_src"]
        out["rules"][k] = {"label": RULES[k], "mean_f1": float(F[k].mean()), "mean_regret": float(reg.mean()), "median_regret": float(np.median(reg)),
                           "max_regret": float(reg.max()), "mean_regret_ci95_cell_bootstrap": ci(reg_b),
                           "gap_closed": gc, "gap_closed_ci95_cell_bootstrap": ci(gc_b[gc_ok]) if gc is not None and gc_ok.sum() > 0 else None,
                           "gap_closed_n_replicates_with_nonpositive_gap": int((~gc_ok).sum() + (gap_b[gc_ok] <= 0).sum()) if gc is not None else None,
                           "mean_f1_gain_vs_t_src": float(d.mean()), "mean_f1_gain_vs_t_src_ci95_cell_bootstrap": ci(d[idx].mean(axis=1)),
                           "n_beats_t_src": int((d > TIE).sum()), "n_worse_than_t_src": int((d < -TIE).sum()), "n_ties_t_src": int((np.abs(d) <= TIE).sum())}
    # paired kept-rate minus quantile-pooled
    dd = F["M1c_kept_rate"] - F["M1a_quantile_pooled"]
    dd_b = dd[idx].mean(axis=1)
    out["paired_kept_rate_minus_quantile_pooled"] = {
        "definition": "per cell F1(M1c kept-rate) - F1(M1a quantile-pooled); positive = kept-rate better (its regret is lower by the same amount)",
        "mean": float(dd.mean()), "median": float(np.median(dd)), "ci95_cell_bootstrap": ci(dd_b),
        "p_two_sided_cell_bootstrap": float(2 * min(np.mean(dd_b <= 0), np.mean(dd_b >= 0))),
        "n_wins_kept_rate": int((dd > TIE).sum()), "n_losses_kept_rate": int((dd < -TIE).sum()), "n_ties": int((np.abs(dd) <= TIE).sum()),
        "mean_regret_kept_rate": float((F["ORC_target_calval"] - F["M1c_kept_rate"]).mean()),
        "mean_regret_quantile_pooled": float((F["ORC_target_calval"] - F["M1a_quantile_pooled"]).mean()),
        "per_cell": {c: float(v) for c, v in zip(sel, dd)}}
    return out


def main():
    t0 = time.time()
    cells, r, p = load_cells()
    names = sorted(cells)
    by_dir = {d: [c for c in names if cells[c]["direction"] == d] for d in DIR_ORDER}
    assert [len(by_dir[d]) for d in DIR_ORDER] == [13, 13, 9, 9], {d: len(v) for d, v in by_dir.items()}
    below = [c for c in names if not cells[c]["oracle_f1_ge_floor"]]
    groups = {}
    for d in DIR_ORDER + ["overall"]:
        sel = by_dir[d] if d != "overall" else names
        groups[d] = {"all": summarise(sel, cells), "oracle_f1_ge_0.10": summarise([c for c in sel if cells[c]["oracle_f1_ge_floor"]], cells)}
    # sanity: overall/all mean regret must equal the refine summary
    rs = r["summary"]["groups"]["all"]
    for k in ("D25_default", "M1c_kept_rate"):
        assert abs(groups["overall"]["all"]["rules"][k]["mean_regret"] - rs[k]["regret_mean"]) < 1e-9, k
    assert abs(groups["overall"]["all"]["rules"]["M1a_quantile_pooled"]["mean_regret"] - p["summary"]["groups"]["all"]["M1a_quantile_pooled"]["regret_mean"]) < 1e-9
    out = {"created": datetime.now().isoformat(timespec="seconds"), "inputs": {"refine": str(REFINE), "pilot": str(PILOT),
           "refine_created": r["created"], "pilot_created": p["created"]},
           "protocol": __doc__, "floor_oracle_f1": FLOOR, "tie_tolerance": TIE, "bootstrap": {"unit": "cell", "B": B, "seed": SEED},
           "directions": {d: {"n_cells": len(by_dir[d]), "cells": by_dir[d]} for d in DIR_ORDER},
           "cells_below_floor": {"n": len(below), "floor": FLOOR,
                                 "cells": {c: {"direction": cells[c]["direction"], "oracle_f1": cells[c]["f1"]["ORC_target_calval"], "f1_t_src": cells[c]["f1"]["M0_t_src"],
                                               "f1_kept_rate": cells[c]["f1"]["M1c_kept_rate"], "f1_quantile_pooled": cells[c]["f1"]["M1a_quantile_pooled"],
                                               "f1_default": cells[c]["f1"]["D25_default"], "n_target_test_gt": cells[c]["n_target_test_gt"]} for c in below}},
           "n_cells_at_or_above_floor": len(names) - len(below),
           "groups": groups, "cells": cells,
           "reproduction_checks": {"pilot_vs_refine_f1_agree_1e-9": True, "overall_all_mean_regret_equals_refine_summary_1e-9": True,
                                   "overall_all_M1a_mean_regret_equals_pilot_summary_1e-9": True},
           "seconds": round(time.time() - t0, 2)}
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ markdown
    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"
    cis = lambda v: "n/a" if v is None else f"[{v[0]:.3f}, {v[1]:.3f}]"
    L = ["# Label-free threshold transfer by direction", "",
         f"Written by `label_free_by_direction.py` on {out['created']} from `label_free_threshold_refine.json` ({r['created']}) and "
         f"`label_free_threshold_pilot.json` ({p['created']}). All numbers: `label_free_by_direction.json`.", "",
         f"Cell-level bootstrap (resample cells with replacement, seed {SEED}, B = {B:,}, percentile 2.5-97.5, same resamples for every rule). "
         f"gap closed = 1 - mean regret / mean gap (ratio of means). Ties: |dF1| <= {TIE:g}. "
         f"Floor: target-calval oracle threshold reaching target-test box F1 >= {FLOOR}.", "",
         f"Cells below the floor: {len(below)} of {len(names)} (so {len(names) - len(below)} cells at or above it) - key `cells_below_floor`.", "",
         "| cell | direction | oracle F1 | F1 t_src | F1 0.25 | F1 kept-rate | F1 quantile-pooled | n test GT |", "|---|---|---|---|---|---|---|---|"]
    for c in below:
        x = out["cells_below_floor"]["cells"][c]
        L.append(f"| {c} | {x['direction']} | {x['oracle_f1']:.3f} | {x['f1_t_src']:.3f} | {x['f1_default']:.3f} | {x['f1_kept_rate']:.3f} | {x['f1_quantile_pooled']:.3f} | {x['n_target_test_gt']} |")
    L.append("")
    for subset, title in (("all", "All cells"), ("oracle_f1_ge_0.10", f"Cells with oracle F1 >= {FLOOR}")):
        L += [f"## {title}", "",
              "| direction | n | mean F1 t_src | mean F1 0.25 | mean F1 kept-rate | mean F1 q-pooled | mean F1 oracle | mean gap [cell CI] |",
              "|---|---|---|---|---|---|---|---|"]
        for d in DIR_ORDER + ["overall"]:
            g = groups[d][subset]
            if g["n_cells"] == 0:
                L.append(f"| {d} | 0 | | | | | | |"); continue
            mf = g["mean_f1"]
            L.append(f"| {d} | {g['n_cells']} | {mf['M0_t_src']:.3f} | {mf['D25_default']:.3f} | {mf['M1c_kept_rate']:.3f} | {mf['M1a_quantile_pooled']:.3f} | "
                     f"{mf['ORC_target_calval']:.3f} | {g['mean_gap_oracle_minus_t_src']:.3f} {cis(g['mean_gap_ci95_cell_bootstrap'])} |")
        L += ["", "| direction | n | rule | mean regret [cell CI] | gap closed [cell CI] | mean dF1 vs t_src [cell CI] | beats / worse / ties vs t_src |",
              "|---|---|---|---|---|---|---|"]
        for d in DIR_ORDER + ["overall"]:
            g = groups[d][subset]
            if g["n_cells"] == 0:
                continue
            for k in RULES:
                v = g["rules"][k]
                L.append(f"| {d} | {g['n_cells']} | {RULES[k]} | {v['mean_regret']:.3f} {cis(v['mean_regret_ci95_cell_bootstrap'])} | "
                         f"{f3(v['gap_closed'])} {cis(v['gap_closed_ci95_cell_bootstrap'])} | {v['mean_f1_gain_vs_t_src']:+.3f} {cis(v['mean_f1_gain_vs_t_src_ci95_cell_bootstrap'])} | "
                         f"{v['n_beats_t_src']} / {v['n_worse_than_t_src']} / {v['n_ties_t_src']} |")
        L += ["", "Paired per-cell difference F1(kept-rate) - F1(quantile-pooled) (positive = kept-rate better):", "",
              "| direction | n | mean regret kept-rate | mean regret q-pooled | mean diff [cell CI] | two-sided cell-bootstrap p | wins / losses / ties (kept-rate) |",
              "|---|---|---|---|---|---|---|"]
        for d in DIR_ORDER + ["overall"]:
            g = groups[d][subset]
            if g["n_cells"] == 0:
                continue
            q = g["paired_kept_rate_minus_quantile_pooled"]
            L.append(f"| {d} | {g['n_cells']} | {q['mean_regret_kept_rate']:.3f} | {q['mean_regret_quantile_pooled']:.3f} | {q['mean']:+.4f} {cis(q['ci95_cell_bootstrap'])} | "
                     f"{q['p_two_sided_cell_bootstrap']:.3f} | {q['n_wins_kept_rate']} / {q['n_losses_kept_rate']} / {q['n_ties']} |")
        L.append("")
    L += ["## Per-cell F1 (target test, box F1 at IoU 0.5)", "",
          "| cell | direction | oracle | t_src | 0.25 | kept-rate | q-pooled | gap | regret kept-rate | regret q-pooled | kept - q-pooled | >= floor |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in names:
        x = cells[c]; f = x["f1"]
        L.append(f"| {c} | {x['direction']} | {f['ORC_target_calval']:.3f} | {f['M0_t_src']:.3f} | {f['D25_default']:.3f} | {f['M1c_kept_rate']:.3f} | {f['M1a_quantile_pooled']:.3f} | "
                 f"{x['gap_oracle_minus_t_src']:.3f} | {x['regret']['M1c_kept_rate']:.3f} | {x['regret']['M1a_quantile_pooled']:.3f} | "
                 f"{f['M1c_kept_rate'] - f['M1a_quantile_pooled']:+.4f} | {'yes' if x['oracle_f1_ge_floor'] else 'no'} |")
    L += ["", f"Keys: `groups.<direction>.<subset>.rules.<rule>.{{mean_regret, mean_regret_ci95_cell_bootstrap, gap_closed, gap_closed_ci95_cell_bootstrap, "
          f"n_beats_t_src, n_worse_than_t_src, n_ties_t_src}}`; `groups.<direction>.<subset>.paired_kept_rate_minus_quantile_pooled`; `cells.<cell>.f1.<rule>`. "
          f"Seconds: {out['seconds']}.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("->", OUT_JSON, OUT_MD, f"{out['seconds']} s")
    for d in DIR_ORDER + ["overall"]:
        for subset in ("all", "oracle_f1_ge_0.10"):
            g = groups[d][subset]
            if g["n_cells"] == 0:
                continue
            kr, qp = g["rules"]["M1c_kept_rate"], g["rules"]["M1a_quantile_pooled"]
            pd_ = g["paired_kept_rate_minus_quantile_pooled"]
            print(f"{d:28s} {subset:18s} n={g['n_cells']:2d} gap {g['mean_gap_oracle_minus_t_src']:.3f} | kept-rate regret {kr['mean_regret']:.3f} {cis(kr['mean_regret_ci95_cell_bootstrap'])} "
                  f"closed {f3(kr['gap_closed'])} {cis(kr['gap_closed_ci95_cell_bootstrap'])} | q-pooled regret {qp['mean_regret']:.3f} closed {f3(qp['gap_closed'])} "
                  f"| paired {pd_['mean']:+.4f} {cis(pd_['ci95_cell_bootstrap'])} W/L/T {pd_['n_wins_kept_rate']}/{pd_['n_losses_kept_rate']}/{pd_['n_ties']}")


if __name__ == "__main__":
    main()
