#!/bin/bash
# Quantum benchmarks of KANO (double well and NLSE) under full, position-and-momentum and position-only supervision (Tables 3-4, Fig. 4).
# Needs data/quantum/doublewell_100.pt and data/quantum/nlse_100.pt (see README). The paper runs used one GPU per scenario.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
export DEVICE=${DEVICE:-cuda} ITERS=5000 LB=800
for ds in dw nlse; do
  for scen in full posmom pos; do
    for s in 0 1 2; do
      $PY -B bench/run_quantum3.py $ds $scen $s dictpos U3
    done
  done
done
# Distillation of the potential and reselection of the Kerr term (NLSE, CPU)
for scen in full posmom pos; do
  for s in 0 1 2; do
    OMP_NUM_THREADS=2 $PY -B bench/quantum_symbolic.py nlse $scen $s
  done
done
