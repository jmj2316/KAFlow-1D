#!/bin/bash
# Every number in the tables and every figure, from the stored results and checkpoints in runs/ (no training).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
$PY -B paper/scripts/compute_numbers.py        # -> paper/data/numbers_G.json
$PY -B paper/scripts/compute_numbers2.py       # -> paper/data/numbers_rest.json
$PY -B paper/scripts/compute_selected.py all   # -> paper/data/selected.json (validation-selected models)
$PY -B paper/scripts/dispersion.py             # -> paper/data/dispersion.json
$PY -B paper/scripts/make_figs.py              # -> paper/figs/fig2.pdf ... fig6.pdf
$PY -B paper/scripts/make_fig1.py              # -> paper/figs/fig1.pdf
