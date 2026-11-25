"""Regenerate the KANO and baseline model classes from a local clone of the official KANO repository.

The official repository (https://github.com/jinucsb/KANO) has no license file, so its code is not redistributed here.
This script copies the model classes verbatim from its notebooks into the two modules that the baseline and
evaluation scripts import:

  ext_kano_official.py   G_3/KANO_G_3.ipynb, code cell 2, up to the training module (matplotlib import removed)
  bench/ext_baselines.py U-FNO, AM-FNO and PDNO classes, first cell of the G_1 notebooks

  git clone https://github.com/jinucsb/KANO ext/KANO
  python tools/extract_kano.py [path/to/KANO]
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KANO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'ext', 'KANO')


def cell(nb, i):
    return ''.join(json.load(open(os.path.join(KANO, nb), encoding='utf-8'))['cells'][i]['source'])


src = '\n'.join(l for l in cell('G_3/KANO_G_3.ipynb', 2).split('\n') if 'matplotlib' not in l)
src = src[:src.index('# Caching-Utilizing Train Module')]
open(os.path.join(ROOT, 'ext_kano_official.py'), 'w', encoding='utf-8').write(
    '"""Official KANO model classes (jinucsb/KANO, G_3/KANO_G_3.ipynb cell 2, copied by tools/extract_kano.py)."""\n' + src)

parts = ['"""Baseline model classes copied from the G_1 notebooks of the official KANO repository by tools/extract_kano.py."""\n'
         'import math, torch, torch.nn as nn, torch.nn.functional as F\n']
for name in ('UFNO', 'AMFNO', 'PDNO'):
    parts.append(f'\n# ======== {name} ========\n' + cell(f'G_1/{name}_G_1.ipynb', 0) + '\n')
open(os.path.join(ROOT, 'bench', 'ext_baselines.py'), 'w', encoding='utf-8').write(''.join(parts))
print('wrote ext_kano_official.py and bench/ext_baselines.py from', KANO)
