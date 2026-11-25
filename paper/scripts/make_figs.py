"""Figures 2–6 of the paper: 178 mm wide, bold uppercase panel labels, no minor ticks, no red-green combinations (colorblind-safe palette), vector PDF.
 python -B paper/scripts/make_figs.py   → paper/figs/fig2..fig6.pdf
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
K = ROOT; OUT = f'{K}/paper/figs'; os.makedirs(OUT, exist_ok=True)
A = json.load(open(f'{K}/paper/data/numbers_G.json')); B = json.load(open(f'{K}/paper/data/numbers_rest.json'))
DSP = json.load(open(f'{K}/paper/data/dispersion.json'))
# overwrite with the single models selected by validation loss (compute_selected.py)
SEL = json.load(open(f'{K}/paper/data/selected.json'))
A = {'family': {g: SEL['G'][g]['family'] for g in ('1', '2', '3')}, 'interp': {g: SEL['G'][g]['interp'] for g in ('1', '2', '3')},
     'slices': {g: SEL['G'][g]['slices'] for g in ('1', '2', '3')}}
for k, v in SEL['Q'].items():
    B['quantum'][k]['curve'] = v['curve']
B['H1'] = {k: v for k, v in SEL['H'].items()}
if 'D' in SEL:
    B['dyn'] = {eq: dict(steps=[1, 10, 25, 50, 100], kaflow=v['kaflow'], fno=v['fno'], pdefind=v['pdefind'], weak=v['weak']) for eq, v in SEL['D'].items()}
MM = 1 / 25.4
plt.rcParams.update({
    'font.family': 'Liberation Sans', 'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.3, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.visible': False, 'ytick.minor.visible': False,
    'lines.linewidth': 1.0, 'savefig.dpi': 400, 'pdf.fonttype': 42, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False, 'mathtext.fontset': 'dejavusans'})
# Okabe–Ito colorblind-safe palette (no red or green)
C = {'KAFlow': '#D55E00', 'KANO': '#0072B2', 'KANO_mlp': '#56B4E9', 'FNO': '#000000', 'PDE-FIND': '#999999', 'Weak SINDy': '#666666',
     'full': '#0072B2', 'posmom': '#D55E00', 'pos': '#CC79A7'}


def plabel(ax, s, x=-0.18, y=1.04):
    ax.text(x, y, s.upper(), transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')


def nominor(ax):
    ax.minorticks_off()


# ---------------- Figure 2: synthetic operators (A–C loss by function family, D–F interpolation)
fams = ['sine_beats', 'chirped_cosine', 'periodic', 'wave_packet', 'sinc_pulse', 'gaussian_hermite']
lab = ['Sine\nbeats', 'Chirped\ncosine', 'Peri-\nodic', 'Wave\npacket', 'Sinc\npulse', 'Gauss.\nHermite']
fig, axs = plt.subplots(2, 3, figsize=(178 * MM, 98 * MM))
for i, g in enumerate(('1', '2', '3')):
    ax = axs[0, i]
    ax.axvspan(2.5, 5.5, color='#eeeeee', zorder=0)
    for j, nm in enumerate(('KAFlow', 'KANO', 'FNO')):
        v = [A['family'][g][nm][f] for f in fams]
        ax.plot(np.arange(6) + (j - 1) * 0.12, v, 'o-', ms=3.2, lw=0.8, color=C[nm], label=nm)
    ax.set_yscale('log'); nominor(ax); ax.set_xlim(-0.5, 5.5)
    ax.set_xticks(range(6)); ax.set_xticklabels(lab, fontsize=5.8)
    ax.set_ylabel(r'Relative $\ell_2$ loss'); ax.set_title(rf'$\mathcal{{G}}_{g}$')
    plabel(ax, 'ABC'[i])
    if i == 0: ax.legend(loc='upper left', fontsize=6)
al = np.linspace(0, 1, 101)
for i, g in enumerate(('1', '2', '3')):
    ax = axs[1, i]
    for nm in ('KAFlow', 'KANO', 'FNO'):
        ax.plot(al, A['interp'][g][nm], '-', lw=1.1, color=C[nm], label=nm)
    ax.axhline(1, color='#bbbbbb', lw=0.5, zorder=0)
    ax.set_xlabel('Interpolation ratio (group A to B)'); ax.set_ylabel('Loss ratio'); nominor(ax)
    plabel(ax, 'DEF'[i])
fig.tight_layout(w_pad=1.2, h_pad=1.0); fig.savefig(f'{OUT}/fig2.pdf'); plt.close(fig)

# ---------------- Figure 3: learned components
S1, S2, S3 = A['slices']['1'], A['slices']['2'], A['slices']['3']
x = np.array(S1['x']); xi = np.array(S1['xi']); band = xi <= 40; inn = np.abs(x) <= 5 * np.pi / 6
fig, axs = plt.subplots(1, 4, figsize=(178 * MM, 42 * MM))
ax = axs[0]; ax.plot(x[inn], np.array(S1['p_xi0_re'])[inn], color=C['KAFlow'], lw=1.6, label='Learned')
ax.plot(x[inn], x[inn] ** 2, 'k--', lw=0.8, label=r'$x^2$'); ax.set_xlabel(r'$x$'); ax.set_ylabel(r'Re $p(x,0)$ of $\mathcal{G}_1$'); ax.legend()
ax = axs[1]
for g, S, ls, col in (('1', S1, '-', C['KAFlow']), ('2', S2, '-', C['KANO']), ('3', S3, ':', C['pos'])):
    ax.plot(xi[band], np.array(S['p_x0_re'])[band], ls, lw=1.4, color=col, label=rf'$\mathcal{{G}}_{g}$')
ax.plot(xi[band], xi[band] ** 2, 'k--', lw=0.6); ax.plot(xi[band], -xi[band] ** 2, 'k--', lw=0.6)
ax.set_xlabel(r'$\xi$'); ax.set_ylabel(r'Re $p(0,\xi)$'); ax.legend()
ax = axs[2]
for g, S, ls, col in (('2', S2, '-', C['KANO']), ('3', S3, ':', C['pos'])):
    ax.plot(x[inn], np.array(S['p_xij_im'])[inn], ls, lw=1.4, color=col, label=rf'$\mathcal{{G}}_{g}$')
ax.plot(x[inn], x[inn], 'k--', lw=0.6, label=r'$x$'); ax.set_xlabel(r'$x$'); ax.set_ylabel(r'Im $p(x,\xi)/\xi$ at $\xi=10$'); ax.legend()
ax = axs[3]; f = np.array(S3['phi_f']); ph = np.array(S3['phi_val'])
cf = np.linalg.lstsq(np.stack([f ** 3, f], 1), ph, rcond=None)[0]
ax.scatter(f, ph, s=1.2, color=C['KAFlow'], alpha=0.5, lw=0, label=r'Learned $\Phi^{nl}$')
ff = np.linspace(f.min(), f.max(), 200); ax.plot(ff, cf[0] * ff ** 3 + cf[1] * ff, 'k--', lw=0.8, label=rf'${cf[0]:.3f}f^3{cf[1]:+.3f}f$')
ax.set_xlabel(r'$f$'); ax.set_ylabel(r'$\Phi^{nl}$ of $\mathcal{G}_3$'); ax.legend(loc='upper left')
for i, ax in enumerate(axs):
    nominor(ax); plabel(ax, 'ABCD'[i], x=-0.3)
fig.tight_layout(w_pad=1.0); fig.savefig(f'{OUT}/fig3.pdf'); plt.close(fig)

# ---------------- Figure 4: quantum infidelity
QK = {'dw': {'full': 6.3e-6, 'posmom': 6.3e-6, 'pos': 4.7e-3, 'fno': 1.5e-2}, 'nlse': {'full': 6.8e-6, 'posmom': 5.6e-6, 'pos': 6.1e-2, 'fno': 1.6e-2}}
fig, axs = plt.subplots(1, 2, figsize=(178 * MM, 58 * MM))
st = np.arange(1, 101)
for i, ds in enumerate(('dw', 'nlse')):
    ax = axs[i]
    ax.axvspan(0, 10, color='#eeeeee', zorder=0, label='Training steps')
    for scen, ls, nm in (('full', '-', 'KAFlow, full'), ('posmom', '--', 'KAFlow, pos & mom'), ('pos', '-.', 'KAFlow, pos')):
        ax.plot(st, np.clip(B['quantum'][f'{ds}_{scen}']['curve'], 1e-16, None), ls, lw=1.3, color=C[scen], label=nm)
    for scen, mk, nm in (('full', 'o', 'Q-KANO, full'), ('posmom', 's', 'Q-KANO, pos & mom'), ('pos', '^', 'Q-KANO, pos'), ('fno', 'x', 'FNO, full')):
        ax.scatter([100], [QK[ds][scen]], marker=mk, s=16, facecolor='none' if mk != 'x' else 'k', edgecolor='k', lw=0.8, zorder=5, label=nm + ' (step 100)')
    ax.set_yscale('log'); nominor(ax); ax.set_xlabel('Time step'); ax.set_ylabel('State infidelity')
    ax.set_title(['Double-well Hamiltonian', 'Nonlinear Schrödinger equation'][i]); plabel(ax, 'AB'[i], x=-0.14)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=6)
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=2.0); fig.savefig(f'{OUT}/fig4.pdf'); plt.close(fig)

# ---------------- Figure 5: variable-order fractional operator
H = B['H1']; Sx = np.array(H['symbol']['x']); Sxi = np.array(H['symbol']['xi']); bd = Sxi <= 48
T = np.array(H['symbol']['true'])[:, bd]; Lr = np.array(H['symbol']['learned'])[:, bd]
fig = plt.figure(figsize=(178 * MM, 52 * MM)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
for i, (M_, ttl) in enumerate(((T, r'Ground truth $|\xi|^{1+0.5\sin x}$'), (Lr, 'Learned by KAFlow ($R=4$)'))):
    ax = fig.add_subplot(gs[i]); im = ax.pcolormesh(Sxi[bd], Sx, np.clip(M_, 1e-1, None), norm=LogNorm(1e-1, T.max()), cmap='viridis', shading='auto', rasterized=True)
    ax.set_xlabel(r'$\xi$'); ax.set_ylabel(r'$x$'); ax.set_title(ttl); cb = plt.colorbar(im, ax=ax, pad=0.02); cb.ax.minorticks_off()
    plabel(ax, 'AB'[i], x=-0.25)
ax = fig.add_subplot(gs[2])
ffs = ('wave_packet', 'sinc_pulse', 'gaussian_hermite')
for j, (nm, key) in enumerate((('KAFlow', 'KAFlow_R4'), ('KANO_mlp', 'KANO_mlp'), ('FNO', 'FNO'), ('KANO', 'KANO'))):
    ax.plot(np.arange(3) + (j - 1.5) * 0.1, [H[key]['fam'][f] for f in ffs], 'o-', ms=3.2, lw=0.8, color=C[nm],
            label={'KANO': 'KANO (published configuration)', 'KANO_mlp': 'KANO with MLP symbol'}.get(nm, nm))
ax.set_yscale('log'); nominor(ax); ax.set_xticks(range(3)); ax.set_xticklabels(['Wave packet', 'Sinc pulse', 'Gaussian Hermite'])
ax.set_ylabel(r'Relative $\ell_2$ loss (unseen)'); ax.legend(fontsize=5.6, ncol=2, loc='upper center', bbox_to_anchor=(0.45, -0.16), columnspacing=0.8); plabel(ax, 'C', x=-0.22)
fig.tight_layout(w_pad=1.0); fig.savefig(f'{OUT}/fig5.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 6: dynamics (A–E rollout errors, F–J learned linear symbols)
EQ = [('burgers', 'Burgers'), ('fburgers', 'Fractional Burgers'), ('bo', 'Benjamin–Ono'), ('kdv', 'Korteweg–de Vries'), ('whitham', 'Whitham')]
TRUE_LAB = {'burgers': r'$-0.02\,\xi^2$', 'fburgers': r'$-0.05\,|\xi|^{3/2}$', 'bo': r'$-\xi|\xi|$', 'kdv': r'$\xi^3$', 'whitham': r'$-\sqrt{\xi\tanh\xi}$'}
PART = {'burgers': 'Re', 'fburgers': 'Re', 'bo': 'Im', 'kdv': 'Im', 'whitham': 'Im'}
fig, axs = plt.subplots(2, 5, figsize=(178 * MM, 76 * MM))
for i, (eq, nm) in enumerate(EQ):
    ax = axs[0, i]; D = B['dyn'][eq]; s = D['steps']
    for key, lb, mk in (('kaflow', 'KAFlow', 'o'), ('fno', 'FNO', 's'), ('pdefind', 'PDE-FIND', '^'), ('weak', 'Weak SINDy', 'v')):
        y = [D[key][str(t)] for t in s]
        if any(v is None or v != v for v in y):
            continue
        ax.plot(s, y, '-', marker=mk, ms=2.8, lw=0.9, color=C[lb], label=lb)
    ax.set_xscale('log'); ax.set_yscale('log'); nominor(ax); ax.set_xlabel('Rollout step'); ax.set_title(nm)
    if i == 0: ax.set_ylabel(r'Relative $\ell_2$ loss')
    plabel(ax, 'ABCDE'[i], x=-0.3)
    ax = axs[1, i]; d = DSP[eq]
    ax.plot(d['xi'], d['learned'], color=C['KAFlow'], lw=1.8, label='KAFlow')
    ax.plot(d['xi'], d['true'], 'k--', lw=0.7, label='Ground truth')
    tx, ty, va = {'kdv': (0.04, 0.92, 'top'), 'whitham': (0.42, 0.85, 'top')}.get(eq, (0.06, 0.08, 'bottom'))
    ax.text(tx, ty, TRUE_LAB[eq], transform=ax.transAxes, fontsize=6.3, va=va, ha='left')
    ax.set_xlabel(r'$\xi$'); ax.set_ylabel(f'{PART[eq]} ' + r'$\ell(\xi)$'); nominor(ax)
    plabel(ax, 'FGHIJ'[i], x=-0.3)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=6.3)
fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.6, h_pad=1.2); fig.savefig(f'{OUT}/fig6.pdf'); plt.close(fig)
print('IC figs done', cf)
