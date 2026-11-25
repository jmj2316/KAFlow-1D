#!/bin/bash
# Reproduction of the published KANO and baseline rows with the official weights, interpolation test,
# separation-rank analysis (Fig. 1C) and cost measurements (Table 7; CPU, same machine and threads).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
$PY -B bench/kano_verify.py
$PY -B bench/verify_baselines.py
for g in 1 2 3; do $PY -B bench/interp_test.py $g; done
$PY -B bench/generality_check.py
$PY -B bench/speed.py
