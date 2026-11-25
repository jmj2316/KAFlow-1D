#!/bin/bash
# KANO synthetic operators G1-G3 (Tables 1-2, Figs. 2-3): training, ablation without the separable term, symbolic distillation.
# CPU runs, about 5 min each with 2 threads.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
export DEVICE=${DEVICE:-cpu} OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
for s in 0 1 2 3 4 5; do
  for g in 1 2 3; do
    $PY -B bench/run_u_static2.py $g $s U2op                      # KAFlow (sparse support only)
    DENSE_CAND=1 $PY -B bench/run_u_static2.py $g $s U2opd        # KAFlow (dense model also a candidate; reported in the paper)
    $PY -B bench/run_u_static.py $g $s U                          # ablation: no position-dependent low-rank term
    $PY -B bench/symbolic_op.py $g $s                             # distillation of the linear part (reads the U2op checkpoint)
  done
  $PY -B bench/symbolic_op_nl.py 3 $s                             # distillation of the cubic nonlinearity of G3
done
