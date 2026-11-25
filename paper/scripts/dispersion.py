"""Learned linear symbol ell(xi) of the dynamics models (best validation seed) and the true dispersion relations → paper/data/dispersion.json, paper/figs/dispersion.pdf
 python -B paper/scripts/dispersion.py
"""
import sys, json, math
import numpy as np
import torch
import os
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
K = ROOT; sys.path.insert(0, K)
from kafu import KAFlowU, spectrum_stats, rate_stat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.weight': 'bold', 'axes.labelweight': 'bold',
                     'axes.titleweight': 'bold', 'font.size': 15, 'grid.linestyle': '--', 'grid.alpha': 0.5})
TRUE = {'burgers': ('re', lambda x: -0.02 * x ** 2, r'$-0.02\,\xi^2$'), 'fburgers': ('re', lambda x: -0.05 * x ** 1.5, r'$-0.05\,|\xi|^{3/2}$'),
        'bo': ('im', lambda x: -x * np.abs(x), r'$-\xi|\xi|$'), 'kdv': ('im', lambda x: x ** 3, r'$\xi^3$'),
        'whitham': ('im', lambda x: -np.sqrt(x * np.tanh(x)), r'$-\sqrt{\xi\tanh\xi}$')}
NAME = {'burgers': 'Burgers', 'fburgers': 'Fractional Burgers', 'bo': 'Benjamin–Ono', 'kdv': 'KdV', 'whitham': 'Whitham'}
out = {}
fig, axs = plt.subplots(1, 5, figsize=(24, 4.2))
for i, eq in enumerate(TRUE):
    best = None
    for s in range(6):
        d = json.load(open(f'{K}/runs/{eq}_U2_s{s}.json'))
        if d['val_sparse'] == d['val_sparse'] and (best is None or d['val_sparse'] < best[0]):
            best = (d['val_sparse'], s, d)
    _, s, d = best
    D = np.load(f'{K}/data/{eq}.npz'); tr = torch.from_numpy(D['train'])[:-32]
    N, L, dt = int(D['N']), float(D['L']), float(D['dt_obs'])
    E, xr = spectrum_stats(tr, L); rate = rate_stat(tr, dt)
    m = KAFlowU(N, L, dt, E, xr, rate=rate)
    sd = torch.load(f'{K}/runs/{eq}_U2_s{s}.pt', map_location='cpu', weights_only=False); m.load_state_dict(sd)
    with torch.no_grad():
        ell = m.ell()[0].double().numpy() if not torch.is_complex(m.ell()) else m.ell()[0].to(torch.complex128).numpy()
    xi = m.xi.numpy(); band = np.arange(len(xi)) <= N // 3
    part, f, lab = TRUE[eq]
    y = ell.real if part == 're' else ell.imag
    out[eq] = dict(seed=s, support=d['support'], ell_terms=d['ell_terms'], xi=xi[band].tolist(), learned=y[band].tolist(), true=f(xi[band]).tolist(),
                   rel_err=float(np.linalg.norm(y[band] - f(xi[band])) / np.linalg.norm(f(xi[band]))))
    ax = axs[i]; ax.plot(xi[band], y[band], color='#FF8C00', lw=3, label='KAFlow')
    ax.plot(xi[band], f(xi[band]), 'k--', lw=1.6, label=lab)
    ax.set_xlabel(r'$\xi$'); ax.set_ylabel(('Re' if part == 're' else 'Im') + r'$\,\ell(\xi)$'); ax.grid(True); ax.legend(fontsize=12)
    ax.set_title(f'({"abcde"[i]})  {NAME[eq]}', fontsize=18, pad=10)
    print(eq, s, d['support'], 'rel err of symbol on resolved band %.2e' % out[eq]['rel_err'], flush=True)
plt.tight_layout(); plt.savefig(f'{K}/paper/figs/dispersion.pdf', bbox_inches='tight')
json.dump(out, open(f'{K}/paper/data/dispersion.json', 'w'))
