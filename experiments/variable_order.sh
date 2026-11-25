#!/bin/bash
# Variable-order fractional operator G4 = Op(|xi|^(1+0.5 sin x)) (Table 5, Fig. 5).
# The data generator writes the operators H1-H3 of the generality test; H1 is G4 in the paper.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
export DEVICE=${DEVICE:-cpu} OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
$PY -B bench/synthH_data.py
for s in 0 1 2; do
  for r in 2 4 8 16; do
    DENSE_CAND=1 KR=$r $PY -B bench/run_u_static2.py H1 $s R${r}d   # KAFlow with R separable terms
  done
  $PY -B bench/kano_official_train.py H1 $s                         # KANO and KANO with an MLP symbol, official recipe
  $PY -B bench/fno_official_train.py H1 $s                          # FNO, official recipe
done
