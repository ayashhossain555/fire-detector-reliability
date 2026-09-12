#!/bin/bash
# Detached refresh daemon: every 10 min, if no export/refresh job is running and
# some calibration cell lacks a CI (i.e. a run landed), run the full refresh +
# figures. Replaces the runner's unreliable fire-and-forget Popen.
A=E:/ayash/projects/research-hub/papers/05-fire-detector-reliability; PY=$A/.venv/Scripts/python.exe; cd $A/analysis || exit 1
while true; do
  busy=$(wmic process where "name='python.exe'" get CommandLine 2>/dev/null | grep -cE "run_matrix.py|bootstrap_ci.py|threshold_decomposition.py|dfire_leakage.py|camera_conformal.py|export_predictions.py|run_calibration.py")
  missing=$(comm -23 <(ls calibration | sort) <(ls calibration_ci | sort) | wc -l)
  if [ "$busy" -eq 0 ] && [ "$missing" -gt 0 ]; then
    echo "[$(date +%FT%T)] refresh: $missing cells without CI"; bash refresh_analyses.sh; $PY make_figures.py > make_figures.log 2>&1; echo "[$(date +%FT%T)] refresh done rc=$?"
  fi
  sleep 600
done
