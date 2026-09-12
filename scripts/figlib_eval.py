"""FIgLib deployment evaluation: image-level alarms, time-to-detection and
camera-disjoint conformal guarantees on a domain none of the models saw.

Stage 1  --score --run <run> [--device 1 --batch 4]
  Runs the trained detector over EVERY FIgLib frame (coco_gt/figlib_all.json,
  protocol settings: imgsz 640, conf 0.001, max_det 300, no TTA) and stores
  per frame: max score (any class), max smoke score, n detections, and the
  top-3 boxes → analysis/figlib_scores/<run>.json. Resumable per sequence.

Stage 2  --analyse
  For every scored run, thresholds = {source F1-optimal t_src (from
  operating_points.json), conformal t_alpha fitted on calibration cameras}:
    * frame-level sensitivity (post-plume frames) and false-alarm rate
      (pre-plume frames), overall and per camera;
    * per sequence: time-to-detection = offset of the first post-plume frame
      with an alarm (minutes; NaN if never), missed-sequence rate, and
      whether any pre-plume frame alarmed (sequence-level false alarm);
    * camera-disjoint split-conformal (4 camera folds, seed 3407): target
      sensitivity 1-alpha on calibration cameras → realised sensitivity on
      held-out cameras vs random-frame split control.
  → analysis/figlib_ttd.json.
Credit: HPWREN (https://www.hpwren.ucsd.edu/).
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GT = HERE / "coco_gt" / "figlib_all.json"
SCORES = HERE / "figlib_scores"
SEED = 3407
ALPHAS = [0.05, 0.10, 0.20]
CANON = {"fire": 1, "smoke": 2}


def score_run(run, device, batch):
    from ultralytics import YOLO
    g = json.loads(GT.read_text(encoding="utf-8"))
    SCORES.mkdir(exist_ok=True)
    outp = SCORES / f"{run}.json"
    done = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else {}
    model = YOLO(str(HERE / "runs" / run / "weights" / "best.pt"))
    names = model.names
    todo = [im for im in g["images"] if str(im["id"]) not in done]
    print(f"[{run}] {len(todo)} frames to score ({len(done)} cached)", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        res = model([im["abs_path"] for im in chunk], conf=0.001, max_det=300, imgsz=640,
                    device=device, verbose=False)
        for im, r in zip(chunk, res):
            sc = r.boxes.conf.tolist() if len(r.boxes) else []
            cl = [names[int(c)] for c in r.boxes.cls.tolist()] if len(r.boxes) else []
            smoke = [s for s, c in zip(sc, cl) if c == "smoke"]
            top = sorted(zip(sc, cl, r.boxes.xyxy.tolist()), key=lambda x: -x[0])[:3] if sc else []
            done[str(im["id"])] = {"max": max(sc, default=0.0), "max_smoke": max(smoke, default=0.0),
                                   "n": len(sc), "top3": [[round(s, 4), c, [round(v, 1) for v in b]] for s, c, b in top]}
        if (i // batch) % 250 == 0:
            outp.write_text(json.dumps(done), encoding="utf-8")
            print(f"  {i + len(chunk)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    outp.write_text(json.dumps(done), encoding="utf-8")
    print(f"[{run}] scored {len(done)} frames -> {outp}")


def conformal_threshold(pos_scores, alpha):
    s = np.sort(np.asarray(pos_scores)); k = max(int(np.floor((len(s) + 1) * alpha)), 1)
    return float(s[k - 1])


def frame_eval(frames, score, t):
    pos = np.array([f["smoke_visible"] for f in frames]); sc = np.array([score[str(f["id"])] for f in frames])
    al = sc >= t
    return {"sensitivity": float(al[pos].mean()) if pos.any() else None,
            "false_alarm_rate": float(al[~pos].mean()) if (~pos).any() else None,
            "n_pos": int(pos.sum()), "n_neg": int((~pos).sum())}


def ttd_eval(frames, score, t):
    seqs = defaultdict(list)
    for f in frames:
        seqs[f["sequence"]].append(f)
    ttd, missed, seq_fa, n = [], 0, 0, 0
    for s, fr in seqs.items():
        fr = sorted(fr, key=lambda x: x["offset_s"])
        post = [f for f in fr if f["offset_s"] >= 0]; pre = [f for f in fr if f["offset_s"] < 0]
        if not post:
            continue
        n += 1
        hit = [f["offset_s"] for f in post if score[str(f["id"])] >= t]
        if hit:
            ttd.append(hit[0] / 60.0)
        else:
            missed += 1
        if any(score[str(f["id"])] >= t for f in pre):
            seq_fa += 1
    return {"n_sequences": n, "missed_frac": missed / n if n else None,
            "ttd_min_median": float(np.median(ttd)) if ttd else None,
            "ttd_min_mean": float(np.mean(ttd)) if ttd else None,
            "ttd_min_p90": float(np.percentile(ttd, 90)) if ttd else None,
            "detected_within_10min_frac": (sum(1 for v in ttd if v <= 10) / n) if n else None,
            "sequence_false_alarm_frac": seq_fa / n if n else None}


def analyse():
    g = json.loads(GT.read_text(encoding="utf-8"))
    frames = g["images"]
    ops = json.loads((HERE / "operating_points.json").read_text(encoding="utf-8"))["cells"]
    cams = sorted({f["camera"] for f in frames})
    counts = defaultdict(int)
    for f in frames:
        counts[f["camera"]] += 1
    folds, sizes = [[] for _ in range(4)], [0] * 4
    for cam, n in sorted(counts.items(), key=lambda x: -x[1]):
        i = int(np.argmin(sizes)); folds[i].append(cam); sizes[i] += n
    out = {"n_frames": len(frames), "n_sequences": len({f["sequence"] for f in frames}), "n_cameras": len(cams),
           "folds": [{"cameras": f, "n_frames": s} for f, s in zip(folds, sizes)], "runs": {}}
    rng = np.random.default_rng(SEED)
    for sp in sorted(SCORES.glob("*.json")):
        run = sp.stem
        raw = json.loads(sp.read_text(encoding="utf-8"))
        if len(raw) < len(frames):
            print(f"[skip] {run}: {len(raw)}/{len(frames)} frames scored"); continue
        key = "max_smoke" if "_dfire_" in run else "max"
        score = {k: v[key] for k, v in raw.items()}
        t_src = next((o["source"]["t_f1opt"] for c, o in ops.items() if c.startswith(run + "__to__")), None)
        r = {"score_used": key, "t_src": t_src, "at_t_src": None, "conformal": {}, "per_camera_at_t_src": {}}
        if t_src is not None:
            r["at_t_src"] = {**frame_eval(frames, score, t_src), **ttd_eval(frames, score, t_src)}
            for cam in cams:
                r["per_camera_at_t_src"][cam] = frame_eval([f for f in frames if f["camera"] == cam], score, t_src)
        # MATCHED false-alarm operating points: threshold = (1-FAR) quantile of
        # pre-plume-frame scores, fitted on all frames (pooled) and on the
        # calibration cameras of each fold (camera-disjoint, evaluated held-out).
        r["at_matched_far"] = {}
        neg_all = np.array([score[str(f["id"])] for f in frames if not f["smoke_visible"]])
        for far in (0.01, 0.05, 0.10):
            t_pool = float(np.quantile(neg_all, 1 - far))
            pooled = {"t": t_pool, **frame_eval(frames, score, t_pool), **ttd_eval(frames, score, t_pool)}
            cdm = []
            for fi in range(4):
                te = [f for f in frames if f["camera"] in folds[fi]]; ca = [f for f in frames if f["camera"] not in folds[fi]]
                t = float(np.quantile([score[str(f["id"])] for f in ca if not f["smoke_visible"]], 1 - far))
                cdm.append({"fold": fi, "t": t, **frame_eval(te, score, t), **ttd_eval(te, score, t)})
            agg2 = lambda k: float(np.mean([x[k] for x in cdm if x[k] is not None]))
            r["at_matched_far"][str(far)] = {"pooled": pooled,
                                             "camera_disjoint": {k: agg2(k) for k in ("sensitivity", "false_alarm_rate", "missed_frac", "ttd_min_median", "detected_within_10min_frac", "sequence_false_alarm_frac")},
                                             "camera_disjoint_folds": cdm}
        for alpha in ALPHAS:
            cd, rd = [], []
            for fi in range(4):
                te = [f for f in frames if f["camera"] in folds[fi]]; ca = [f for f in frames if f["camera"] not in folds[fi]]
                t = conformal_threshold([score[str(f["id"])] for f in ca if f["smoke_visible"]], alpha)
                cd.append({"fold": fi, "t": t, **frame_eval(te, score, t), **ttd_eval(te, score, t)})
                for _ in range(10):
                    perm = rng.permutation(len(frames)); te_r = [frames[i] for i in perm[:len(te)]]; ca_r = [frames[i] for i in perm[len(te):]]
                    t_r = conformal_threshold([score[str(f["id"])] for f in ca_r if f["smoke_visible"]], alpha)
                    rd.append(frame_eval(te_r, score, t_r))
            agg = lambda lst, k: {"mean": float(np.mean([x[k] for x in lst if x[k] is not None])), "min": float(np.min([x[k] for x in lst if x[k] is not None]))}
            r["conformal"][str(alpha)] = {"target_sensitivity": 1 - alpha,
                                          "camera_disjoint": {"sensitivity": agg(cd, "sensitivity"), "false_alarm_rate": agg(cd, "false_alarm_rate"),
                                                              "ttd_min_median": float(np.nanmedian([x["ttd_min_median"] if x["ttd_min_median"] is not None else np.nan for x in cd])), "folds": cd},
                                          "random_split": {"sensitivity": agg(rd, "sensitivity"), "false_alarm_rate": agg(rd, "false_alarm_rate")}}
        out["runs"][run] = r
        a = r["at_t_src"] or {}
        print(f"{run}: t_src {t_src} sens {a.get('sensitivity')} FAR {a.get('false_alarm_rate')} TTD med {a.get('ttd_min_median')} min missed {a.get('missed_frac')}")
    (HERE / "figlib_ttd.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("artefact ->", HERE / "figlib_ttd.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true"); ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--run"); ap.add_argument("--device", default="1"); ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    if a.score:
        score_run(a.run, a.device, a.batch)
    if a.analyse:
        analyse()


if __name__ == "__main__":
    main()
