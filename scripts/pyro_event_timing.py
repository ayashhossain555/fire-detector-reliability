"""Time-resolved alarms on Pyro-SDIS val events (uses existing detections).

Frames are grouped into EVENTS per camera (a gap > 10 min between
consecutive frames starts a new event). Within an event with >= 1 annotated
smoke frame, every frame gets a time offset from the FIRST annotated frame
(onset). Frames before onset are the "negatives" that the image-level
false-alarm rate counts against the model. For each model and threshold
(source F1-optimal t_src from operating_points.json) this reports the alarm
rate of negative frames by their distance to onset (pre-onset <=2 min,
2-10 min, >10 min; post-window negatives), the alarm rate of annotated
frames, and per-event "lead time" = minutes between the first alarm and
annotated onset (negative = model fired before the annotators' first box).
Writes analysis/pyro_sdis_event_timing.json.
"""
import collections
import datetime as dt
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "datasets" / "pyro_sdis_image_meta.json").read_text(encoding="utf-8"))
RUNS = ["v8s_pyrosdis_s3407", "y11s_pyrosdis_s3407", "rtdetrl_pyrosdis_s3407",
        "v8s_dfire_s3407", "y11s_dfire_s3407", "rtdetrl_dfire_s3407"]
GAP_S = 600


def load():
    frames = {}
    for tag in ("pyro_sdis_calval", "pyro_sdis_caltest"):
        g = json.loads((HERE / "coco_gt" / f"{tag}.json").read_text(encoding="utf-8"))
        pid = {a["image_id"] for a in g["annotations"]}
        for im in g["images"]:
            frames[im["file_name"]] = {"tag": tag, "id": im["id"], "pos": im["id"] in pid,
                                       "camera": META[im["file_name"]]["camera"],
                                       "t": dt.datetime.strptime(META[im["file_name"]]["date"], "%Y-%m-%dT%H-%M-%S")}
    return frames


def scores_for(run, frames):
    sc = {}
    for tag in ("pyro_sdis_calval", "pyro_sdis_caltest"):
        p = HERE / "detections" / f"{run}__{tag}.bbox.json"
        if not p.exists():
            return None
        best = collections.defaultdict(float)
        for d in json.loads(p.read_text(encoding="utf-8")):
            best[d["image_id"]] = max(best[d["image_id"]], d["score"])
        for f, v in frames.items():
            if v["tag"] == tag:
                sc[f] = best.get(v["id"], 0.0)
    return sc


def events(frames):
    bycam = collections.defaultdict(list)
    for f, v in frames.items():
        bycam[v["camera"]].append((v["t"], f))
    ev = []
    for cam, lst in bycam.items():
        lst.sort()
        cur = [lst[0]]
        for a, b in zip(lst, lst[1:]):
            if (b[0] - a[0]).total_seconds() > GAP_S:
                ev.append(cur); cur = []
            cur.append(b)
        ev.append(cur)
    return ev


def main():
    frames = load()
    ev = events(frames)
    ops = json.loads((HERE / "operating_points.json").read_text(encoding="utf-8"))["cells"]
    out = {"gap_s": GAP_S, "n_events": len(ev), "n_events_with_smoke": sum(any(frames[f]["pos"] for _, f in e) for e in ev),
           "events_frames_median": float(np.median([len(e) for e in ev])), "runs": {}}
    for run in RUNS:
        sc = scores_for(run, frames)
        if sc is None:
            continue
        t = next((o["source"]["t_f1opt"] for c, o in ops.items() if c.startswith(run + "__to__")), None)
        if t is None:
            continue
        bins = collections.defaultdict(lambda: [0, 0])  # [alarms, frames]
        lead, missed = [], 0
        for e in ev:
            pos_t = [frames[f]["t"] for _, f in e if frames[f]["pos"]]
            if not pos_t:
                for _, f in e:
                    bins["event_without_any_annotation"][0] += sc[f] >= t; bins["event_without_any_annotation"][1] += 1
                continue
            onset, last = min(pos_t), max(pos_t)
            first_alarm = min((frames[f]["t"] for _, f in e if sc[f] >= t), default=None)
            if first_alarm is None:
                missed += 1
            else:
                lead.append((first_alarm - onset).total_seconds() / 60.0)
            for _, f in e:
                v = frames[f]
                if v["pos"]:
                    key = "annotated"
                else:
                    d = (onset - v["t"]).total_seconds()
                    if v["t"] < onset:
                        key = "pre_onset_<=2min" if d <= 120 else "pre_onset_2-10min" if d <= 600 else "pre_onset_>10min"
                    elif v["t"] <= last:
                        key = "inside_window_unannotated"
                    else:
                        key = "post_window"
                bins[key][0] += sc[f] >= t; bins[key][1] += 1
        out["runs"][run] = {"t_src": t,
                            "alarm_rate_by_frame_type": {k: {"alarms": int(a), "frames": n, "rate": round(a / n, 4)} for k, (a, n) in bins.items()},
                            "events_with_smoke": len(lead) + missed, "events_missed": missed,
                            "lead_time_min": {"median": float(np.median(lead)) if lead else None,
                                              "frac_alarm_before_annotated_onset": float(np.mean([v < 0 for v in lead])) if lead else None,
                                              "p10": float(np.percentile(lead, 10)) if lead else None,
                                              "p90": float(np.percentile(lead, 90)) if lead else None}}
        r = out["runs"][run]["alarm_rate_by_frame_type"]
        print(run, "t", round(t, 3), {k: v["rate"] for k, v in r.items()}, "lead med", out["runs"][run]["lead_time_min"]["median"],
              "before-onset frac", out["runs"][run]["lead_time_min"]["frac_alarm_before_annotated_onset"], "missed", missed)
    (HERE / "pyro_sdis_event_timing.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("artefact ->", HERE / "pyro_sdis_event_timing.json")


if __name__ == "__main__":
    main()
