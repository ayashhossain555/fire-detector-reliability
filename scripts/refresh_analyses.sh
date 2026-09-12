#!/bin/bash
# Idempotent refresh of every CPU-side analysis after new detections/cells
# land (new runs, new exports). Safe to rerun any time the GPUs are busy.
# Usage: bash analysis/refresh_analyses.sh [--wait-dev1]   (waits for a
#        running "run_matrix.py --device 1" batch first)
A=E:/ayash/projects/research-hub/papers/05-fire-detector-reliability; PY=$A/.venv/Scripts/python.exe; cd $A/analysis || exit 1
if [ "$1" = "--wait-dev1" ]; then
  while wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -qE "run_matrix.py --device 1|bootstrap_ci.py|threshold_decomposition.py|dfire_leakage.py"; do sleep 120; done
fi
echo "[$(date +%FT%T)] refresh start"
for s in dfire_leakage.py threshold_decomposition.py camera_conformal.py operating_points.py pyro_event_timing.py; do
  $PY $s > ${s%.py}.log 2>&1; echo "[$(date +%FT%T)] $s rc=$?"
done
$PY figlib_eval.py --analyse > figlib_analyse.log 2>&1; echo "[$(date +%FT%T)] figlib_eval --analyse rc=$?"
$PY bootstrap_ci.py --all --B 1000 > bootstrap_ci_refresh.log 2>&1; echo "[$(date +%FT%T)] bootstrap_ci rc=$?"
$PY summarise_matrix.py > summarise_matrix.log 2>&1; echo "[$(date +%FT%T)] summarise rc=$? REFRESH DONE"
