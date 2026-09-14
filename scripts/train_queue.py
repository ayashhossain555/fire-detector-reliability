"""Sequential training queue for Paper 05 â€” keeps the single GPU busy
unattended. Jobs are read from analysis/queue.json at EVERY iteration, so
the list can be edited while the queue runs. Each job runs in its own
subprocess (train_run.py) so memory is released between runs.

Idempotent / crash-safe:
  - runs/<name>.done exists            -> skip
  - a live train_run.py for <name>     -> adopt: wait for it, then mark done
                                          (results.png present) or resume
  - runs/<name>/weights/last.pt exists -> resume from last.pt
  - otherwise                          -> fresh start (standard config)
Launch detached (PowerShell, from the paper folder):
  Start-Process .venv/Scripts/python.exe -ArgumentList "analysis/train_queue.py"
    -WindowStyle Hidden -RedirectStandardOutput analysis/runs/queue_stdout.log
    -RedirectStandardError analysis/runs/queue_stderr.log
Progress log: analysis/runs/queue.log (appended by this script).
"""
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
RUNS = HERE / "runs"
PY = PAPER / ".venv" / "Scripts" / "python.exe"
QUEUE_FILE = HERE / "queue.json"
LOG = RUNS / "queue.log"
DATA = {
    "dfire":    PAPER / "data" / "d_fire_yolo" / "data.yaml",
    "pyrosdis": PAPER / "data" / "pyro_sdis_yolo" / "data.yaml",
    "dfiredd":  PAPER / "data" / "d_fire_dedup_yolo" / "data.yaml",   # de-duplicated D-Fire (make_dfire_dedup.py)
    "dfiresub": PAPER / "data" / "d_fire_sub_yolo" / "data.yaml",     # size-matched random-subsample control (make_dfire_sub.py)
}
MODEL = {"v8n": "yolov8n.pt", "v8s": "yolov8s.pt", "v8m": "yolov8m.pt",
         "y11n": "yolo11n.pt", "y11s": "yolo11s.pt", "y11m": "yolo11m.pt",
         "rtdetrl": "rtdetr-l.pt", "rtdetrx": "rtdetr-x.pt"}

def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def completed_epochs(run_dir):
    f = run_dir / "results.csv"
    return max(0, sum(1 for _ in f.open()) - 1) if f.exists() else 0

def live_train_pid(name):
    """PID of a running train_run.py whose --name is `name`, else None."""
    try:
        out = subprocess.run(["wmic", "process", "where", "name='python.exe'",
                              "get", "ProcessId,CommandLine", "/format:csv"],
                             capture_output=True, text=True).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if "train_run.py" in line and f"--name {name}" in line:
            try:
                return int(line.strip().split(",")[-1])
            except ValueError:
                pass
    return None

def next_job():
    jobs = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))["queue"]
    for j in jobs:
        name = f"{j['family']}_{j['dataset']}_s{j['seed']}{j.get('tag', '')}"
        if not (RUNS / f"{name}.done").exists():
            return name, j
    return None, None

def post_process(name):
    """After a run finishes: run-meta JSON, then its matrix cells on the now
    idle training GPU (device 0) â€” minutes there vs hours on the T600."""
    for cmd in ([str(PY), str(HERE / "write_runs_meta.py"), name],
                [str(PY), str(HERE / "run_matrix.py"), "--device", "0", "--only", name]):
        with open(RUNS / f"{name}_post.log", "a") as out:
            rc = subprocess.call(cmd, cwd=str(PAPER), stdout=out, stderr=subprocess.STDOUT)
        log(f"post {name}: {Path(cmd[1]).name} rc={rc}")
    # CPU-side refresh (CIs, decomposition, leakage, summary, figures) runs
    # DETACHED so the next training starts immediately.
    subprocess.Popen(["bash", "-c", f"bash {HERE.as_posix()}/refresh_analyses.sh > {HERE.as_posix()}/refresh_after_{name}.log 2>&1; "
                                    f"{PY.as_posix()} {HERE.as_posix()}/make_figures.py >> {HERE.as_posix()}/refresh_after_{name}.log 2>&1"],
                     cwd=str(PAPER), creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    log(f"post {name}: refresh_analyses + make_figures launched detached")

def mark_done(name, run_dir, wall, note=""):
    (RUNS / f"{name}.done").write_text(json.dumps({
        "finished": datetime.now().isoformat(), "wall_s": round(wall, 1),
        "epochs": completed_epochs(run_dir), "note": note}))

def main():
    log("queue runner started")
    while True:
        name, j = next_job()
        if name is None:
            log("queue complete"); return
        run_dir = RUNS / name
        pid = live_train_pid(name)
        if pid:
            log(f"adopt {name}: live training PID {pid}, waiting")
            t0 = time.time()
            while live_train_pid(name):
                time.sleep(60)
            if (run_dir / "results.png").exists():
                mark_done(name, run_dir, time.time() - t0, "adopted")
                log(f"done {name} (adopted; epochs={completed_epochs(run_dir)})")
                post_process(name)
                continue
            log(f"adopted {name} exited without results.png -> resume")
        args = [str(PY), str(HERE / "train_run.py"), "--name", name,
                "--model", MODEL[j["family"]], "--data", str(DATA[j["dataset"]]),
                "--seed", str(j["seed"]),
                "--batch", str(j.get("batch", 16)), "--workers", str(j.get("workers", 2)),
                "--patience", str(j.get("patience", 15)), "--epochs", str(j.get("epochs", 60))]
        last = run_dir / "weights" / "last.pt"
        if last.exists() and completed_epochs(run_dir) > 0:
            log(f"resume {name} from epoch {completed_epochs(run_dir)}")
            args.append("--resume")
        else:
            log(f"start {name}")
        t0 = time.time()
        with open(RUNS / f"{name}_launch.log", "a") as out, \
             open(RUNS / f"{name}_launch.log.err", "a") as err:
            rc = subprocess.call(args, cwd=str(PAPER), stdout=out, stderr=err)
        dt = time.time() - t0
        if rc == 0:
            mark_done(name, run_dir, dt)
            log(f"done {name} rc=0 wall={dt/3600:.2f} h epochs={completed_epochs(run_dir)}")
            post_process(name)
        else:
            log(f"FAILED {name} rc={rc} after {dt/3600:.2f} h - stopping queue "
                f"(fix, then relaunch; it will resume)")
            sys.exit(rc)

if __name__ == "__main__":
    main()
