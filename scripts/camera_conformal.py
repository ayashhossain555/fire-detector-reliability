"""Camera-disjoint reliability on Pyro-SDIS: do calibration and conformal
alarm guarantees survive a change of camera?

Pyro-SDIS val (4,099 images, 24 cameras, metadata in
datasets/pyro_sdis_image_meta.json) is pooled (calval + caltest) and split by
CAMERA into 4 balanced folds. Everything below uses the already-exported
detections (conf 0.001), so no inference is re-run.

Part 1 — conformal image-level alarm (split conformal, exchangeability):
  image score = max detection score; positive image = has >= 1 smoke box.
  On the calibration cameras take the threshold t_alpha = the k-th smallest
  positive-image score with k = floor((n+1)*alpha), which guarantees
  sensitivity >= 1-alpha on exchangeable test images. Evaluate the realised
  sensitivity and false-alarm rate on (a) held-out CAMERAS (camera-disjoint)
  and (b) a RANDOM image split of the same sizes (exchangeable control,
  20 repeats). Gap (b)-(a) = what camera shift costs the guarantee.
  Run for the Pyro-SDIS-trained models (in-domain) and the D-Fire-trained
  models (cross-dataset alarm: any class fires the alarm).

Part 2 — camera-disjoint post-hoc calibration (LaECE_0 / D-ECE via the
  toolbox): fit temperature/Platt/isotonic on the calibration cameras,
  evaluate on the held-out cameras, vs the same on a random split.

Writes analysis/camera_conformal.json (+ temp GT/det subset files under
analysis/tmp_camera_folds/, deterministic, seed 3407).
"""
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
GT = HERE / "coco_gt"
DETS = HERE / "detections"
TMP = HERE / "tmp_camera_folds"
META = json.loads((HERE / "datasets" / "pyro_sdis_image_meta.json").read_text(encoding="utf-8"))
SEED = 3407
ALPHAS = [0.05, 0.10, 0.20]
N_FOLDS = 4
RUNS = ["v8s_pyrosdis_s3407", "y11s_pyrosdis_s3407", "rtdetrl_pyrosdis_s3407", "rtdetrl_pyrosdis_s3407_p60",
        "v8s_dfire_s3407", "y11s_dfire_s3407", "rtdetrl_dfire_s3407"]


def _q(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def pool():
    """Merge calval + caltest into one image table with unique ids."""
    images, anns, offset = [], [], 0
    for tag in ("pyro_sdis_calval", "pyro_sdis_caltest"):
        g = json.loads((GT / f"{tag}.json").read_text(encoding="utf-8"))
        for im in g["images"]:
            im = dict(im); im["orig_tag"], im["orig_id"] = tag, im["id"]; im["id"] += offset
            im["camera"] = META[im["file_name"]]["camera"]
            images.append(im)
        for a in g["annotations"]:
            a = dict(a); a["image_id"] += offset; a["id"] += offset; anns.append(a)  # ids must stay unique after pooling
        offset += 100000
        cats = g["categories"]
    return images, anns, cats


def load_run_dets(run, images):
    out = {}
    for tag in ("pyro_sdis_calval", "pyro_sdis_caltest"):
        p = DETS / f"{run}__{tag}.bbox.json"
        if not p.exists():
            return None
        off = 0 if tag == "pyro_sdis_calval" else 100000
        for d in json.loads(p.read_text(encoding="utf-8")):
            d = dict(d); d["image_id"] += off
            out.setdefault(d["image_id"], []).append(d)
    return out


def camera_folds(images):
    counts = defaultdict(int)
    for im in images:
        counts[im["camera"]] += 1
    folds = [[] for _ in range(N_FOLDS)]
    sizes = [0] * N_FOLDS
    for cam, n in sorted(counts.items(), key=lambda x: -x[1]):
        i = int(np.argmin(sizes)); folds[i].append(cam); sizes[i] += n
    return folds, sizes


def conformal_threshold(pos_scores, alpha):
    s = np.sort(np.asarray(pos_scores))
    n = len(s)
    k = int(np.floor((n + 1) * alpha))  # k-th smallest (1-indexed); k=0 -> min
    k = max(k, 1)
    return float(s[k - 1])


def alarm_eval(scores, pos, t):
    scores, pos = np.asarray(scores), np.asarray(pos, dtype=bool)
    alarm = scores >= t
    return {"sensitivity": float(alarm[pos].mean()) if pos.any() else None,
            "false_alarm_rate": float(alarm[~pos].mean()) if (~pos).any() else None,
            "n_pos": int(pos.sum()), "n_neg": int((~pos).sum()), "t": t}


def part1(run, images, anns, dets, folds, rng):
    pos_img = {a["image_id"] for a in anns}
    score = {im["id"]: max([d["score"] for d in dets.get(im["id"], [])], default=0.0) for im in images}
    cam_of = {im["id"]: im["camera"] for im in images}
    ids = np.array([im["id"] for im in images])
    res = {"per_alpha": {}, "per_camera_at_alpha0.10": {}}
    for alpha in ALPHAS:
        cd, rd = [], []
        for f in range(N_FOLDS):
            test_cams = set(folds[f])
            te = [i for i in ids if cam_of[i] in test_cams]
            ca = [i for i in ids if cam_of[i] not in test_cams]
            t = conformal_threshold([score[i] for i in ca if i in pos_img], alpha)
            e = alarm_eval([score[i] for i in te], [i in pos_img for i in te], t)
            e["fold"] = f; e["test_cameras"] = sorted(test_cams); cd.append(e)
            # random split control of identical sizes
            for r in range(20):
                perm = rng.permutation(ids)
                te_r, ca_r = perm[:len(te)], perm[len(te):]
                t_r = conformal_threshold([score[i] for i in ca_r if i in pos_img], alpha)
                rd.append(alarm_eval([score[i] for i in te_r], [i in pos_img for i in te_r], t_r))
        def agg(lst, key):
            v = [x[key] for x in lst if x[key] is not None]
            return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} if v else None
        res["per_alpha"][str(alpha)] = {
            "target_sensitivity": 1 - alpha,
            "camera_disjoint": {"sensitivity": agg(cd, "sensitivity"), "false_alarm_rate": agg(cd, "false_alarm_rate"),
                                 "folds": cd},
            "random_split": {"sensitivity": agg(rd, "sensitivity"), "false_alarm_rate": agg(rd, "false_alarm_rate")},
        }
        if alpha == 0.10:
            # per-camera behaviour at the fold thresholds
            for e in cd:
                for cam in e["test_cameras"]:
                    ii = [i for i in ids if cam_of[i] == cam]
                    res["per_camera_at_alpha0.10"][cam] = alarm_eval([score[i] for i in ii], [i in pos_img for i in ii], e["t"])
    return res


def write_subset(name, images, anns, cats, dets, keep_ids):
    TMP.mkdir(exist_ok=True)
    keep = set(keep_ids)
    g = {"images": [im for im in images if im["id"] in keep],
         "annotations": [a for a in anns if a["image_id"] in keep], "categories": cats}
    gp, dp = TMP / f"{name}_gt.json", TMP / f"{name}_dets.json"
    gp.write_text(json.dumps(g), encoding="utf-8")
    dp.write_text(json.dumps([d for i in keep_ids for d in dets.get(i, [])]), encoding="utf-8")
    return str(gp), str(dp)


def part2(run, images, anns, cats, dets, folds, rng):
    from detection_calibration.DetectionCalibration import DetectionCalibration
    import calib_metrics
    ids = np.array([im["id"] for im in images])
    cam_of = {im["id"]: im["camera"] for im in images}
    out = {"camera_disjoint": [], "random_split": []}
    for f in range(N_FOLDS):
        test_cams = set(folds[f])
        splits = {"camera_disjoint": ([i for i in ids if cam_of[i] not in test_cams], [i for i in ids if cam_of[i] in test_cams])}
        perm = rng.permutation(ids); n_te = len(splits["camera_disjoint"][1])
        splits["random_split"] = (list(perm[n_te:]), list(perm[:n_te]))
        for kind, (ca, te) in splits.items():
            cg, cdp = write_subset(f"{run}_{kind}_f{f}_cal", images, anns, cats, dets, ca)
            tg, tdp = write_subset(f"{run}_{kind}_f{f}_test", images, anns, cats, dets, te)
            row = {"fold": f, "n_cal": len(ca), "n_test": len(te)}
            for cal in ("identity", "temperature_scaling", "platt_scaling", "isotonic_regression"):
                cm = DetectionCalibration(cg, tg)
                calibrator, thresholds = _q(cm.fit, cdp, calibrator_type=cal)
                cd = _q(cm.transform, tdp, calibrator, thresholds)
                m = calib_metrics.evaluate(tg, cd)
                row[cal] = {"LaECE_0": m["LaECE_0"], "D_ECE": m["D_ECE"], "lrp": m["lrp"]}
            out[kind].append(row)
    summary = {}
    for kind in out:
        summary[kind] = {cal: {met: float(np.mean([r[cal][met] for r in out[kind]]))
                               for met in ("LaECE_0", "D_ECE")}
                         for cal in ("identity", "temperature_scaling", "platt_scaling", "isotonic_regression")}
    return {"folds": out, "mean_over_folds": summary}


def main():
    images, anns, cats = pool()
    folds, sizes = camera_folds(images)
    result = {"seed": SEED, "n_images": len(images), "n_positive": len({a["image_id"] for a in anns}),
              "n_cameras": len({im["camera"] for im in images}),
              "folds": [{"cameras": f, "n_images": s} for f, s in zip(folds, sizes)],
              "alphas": ALPHAS, "runs": {}}
    for run in RUNS:
        dets = load_run_dets(run, images)
        if dets is None:
            print(f"[skip] {run}: detections on pyro_sdis calval/caltest missing"); continue
        rng = np.random.default_rng(SEED)
        r = {"conformal_alarm": part1(run, images, anns, dets, folds, rng)}
        if "pyrosdis" in run:
            r["posthoc_calibration"] = part2(run, images, anns, cats, dets, folds, np.random.default_rng(SEED))
        result["runs"][run] = r
        pa = r["conformal_alarm"]["per_alpha"]["0.1"]
        print(f"{run}: alpha=0.10 camera-disjoint sens {pa['camera_disjoint']['sensitivity']['mean']:.3f} "
              f"(min {pa['camera_disjoint']['sensitivity']['min']:.3f}) FAR {pa['camera_disjoint']['false_alarm_rate']['mean']:.3f} | "
              f"random sens {pa['random_split']['sensitivity']['mean']:.3f} FAR {pa['random_split']['false_alarm_rate']['mean']:.3f}")
        if "posthoc_calibration" in r:
            s = r["posthoc_calibration"]["mean_over_folds"]
            print("   LaECE_0 camera-disjoint:", {k: round(v["LaECE_0"] * 100, 1) for k, v in s["camera_disjoint"].items()},
                  "| random:", {k: round(v["LaECE_0"] * 100, 1) for k, v in s["random_split"].items()})
    (HERE / "camera_conformal.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("artefact ->", HERE / "camera_conformal.json")


if __name__ == "__main__":
    main()
