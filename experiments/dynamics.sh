#!/bin/bash
# Evolution equations: Burgers, fractional Burgers, Benjamin-Ono, KdV and Whitham (Table 6, Fig. 6).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
export DEVICE=${DEVICE:-cuda}
EQS="burgers fburgers bo kdv whitham"
$PY -B pde1d.py $EQS                                               # data/{eq}.npz (fixed seeds)
for eq in $EQS; do
  for s in 0 1 2 3 4 5; do
    $PY -B train_u2.py $eq $s U2                                   # KAFlow
  done
  $PY -B fno_tune.py sweep $eq                                     # FNO: width x modes x learning-rate sweep on the validation split
  for s in 0 1 2; do
    SAVE_CKPT=1 $PY -B fno_tune.py final $eq $s                    # FNO: best setting of the sweep
  done
  # PDE-FIND and Weak SINDy (PySINDy 1.7.5; see README for its environment)
  OMP_NUM_THREADS=2 $PY -B bench/sindy_baseline.py $eq 0
done
