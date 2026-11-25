"""Figure 1 (concept figure with real-data thumbnails and a clean flow, 178 mm).
 A learning and distillation flow (Benjamin–Ono example: observed trajectories → low-rank generator → exact time integration → automatic distillation → closed form)
 B dense symbol (KANO) vs low-rank symbol (KAFlow), the actual symbols of G2
 C singular-value decay of the generator vs the propagator (computed)
 python -B paper/scripts/make_fig1.py
"""
import sys, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import os
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
K = ROOT; sys.path.insert(0, K)
from kafu import KAFlowU, KAFlowUOp2, spectrum_stats, rate_stat

MM = 1 / 25.4
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 7, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5,
                     'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.labelsize': 5.5,
                     'ytick.labelsize': 5.5, 'xtick.minor.visible': False, 'ytick.minor.visible': False, 'pdf.fonttype': 42,
                     'mathtext.fontset': 'dejavusans', 'axes.spines.top': False, 'axes.spines.right': False})
ORANGE, BLUE, SKY, PINK, GREY, DARK = '#D55E00', '#0072B2', '#56B4E9', '#CC79A7', '#BBBBBB', '#333333'
FILL_K, FILL_Y = '#FDF1E8', '#EEF4FA'

# ---------------- data and models
D = np.load(f'{K}/data/bo.npz'); N, L, dt = int(D['N']), float(D['L']), float(D['dt_obs'])
tr = torch.from_numpy(D['train'])[:-32]; te = D['test']
E, xr = spectrum_stats(tr, L); rate = rate_stat(tr, dt)
mbo = KAFlowU(N, L, dt, E, xr, rate=rate); mbo.load_state_dict(torch.load(f'{K}/runs/bo_U2_s1.pt', map_location='cpu', weights_only=False)); mbo.eval()
with torch.no_grad():
    pred = mbo(torch.from_numpy(te[0:1, 0]), 100)[0].numpy()
    ell_bo = mbo.ell()[0].to(torch.complex128).numpy(); xi_bo = mbo.xi.numpy()
    dsym = mbo.d()[0].to(torch.complex128).numpy()
    ug = torch.linspace(-2.5, 2.5, 200)
    feats = torch.cat([ug[None, :, None], torch.zeros(1, 200, mbo.J)], -1)
    phi_bo = mbo.phi(feats).numpy()[0]
d2 = torch.load(f'{K}/ext/KANO/G_2/dataset/train_2.pt', weights_only=False)
F2 = torch.stack([e[1] for e in d2]).float(); G2 = torch.stack([e[2] for e in d2]).float()
E2, x2 = spectrum_stats(F2, 2 * math.pi); sc = float(G2.pow(2).mean().sqrt() / F2.pow(2).mean().sqrt())
G2SEED = json.load(open(f'{K}/paper/data/selected.json'))['G']['2']['seed']
mg = KAFlowUOp2(128, 2 * math.pi, E2, x2, sc, R=2, J=1, H=2); mg.load_state_dict(torch.load(f'{K}/runs/synth/G2_U2opd_s{G2SEED}.pt', map_location='cpu', weights_only=False)); mg.eval()
with torch.no_grad():
    ar = (mg.ar_scale * mg.ar()).numpy(); mr = mg.mr().to(torch.complex128).numpy(); xg = mg.x.numpy(); xig = mg.xi.numpy()
    Ksum = (ar[:, :, None] * mr[:, None, :]).sum(0)                # sum_r a_r(x) m_r(xi): identifiable (factors are not)
band = np.arange(len(xi_bo)) <= N // 3

fig = plt.figure(figsize=(178 * MM, 128 * MM))


def label(x, y, s):
    fig.text(x, y, s, fontsize=9, fontweight='bold', va='top', ha='left')


# ================= A: flow diagram =================
axA = fig.add_axes([0, 0.50, 1, 0.46]); axA.set_xlim(0, 178); axA.set_ylim(0, 58); axA.axis('off')
label(0.005, 0.985, 'A')


def box(x, y, w, h, fc, ec='#999999', lw=0.6):
    axA.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.6', fc=fc, ec=ec, lw=lw, zorder=0))


def arrow(x0, y0, x1, y1, rad=0.0):
    axA.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=0.8, color=DARK, connectionstyle=f'arc3,rad={rad}'))


def inset(x, y, w, h):
    return axA.inset_axes([x, y, w, h], transform=axA.transData)


def heat(ax, Z, cmap='RdBu_r'):
    v = np.abs(Z).max(); ax.imshow(Z.T, origin='lower', aspect='auto', cmap=cmap, vmin=-v, vmax=v, extent=(0, 1, 0, 1), interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.5); s.set_color('#666666')


# A1 observed trajectories (stacked cards)
axA.text(15, 55.5, 'Observed trajectories', ha='center', fontsize=7, fontweight='bold')
for j, k in enumerate((3, 2, 1)):
    a = inset(4 + 2.2 * j, 20 - 2.2 * j + 4.4, 20, 24); heat(a, D['train'][k])
a = inset(8.4, 15.6, 20, 24); heat(a, D['train'][0])
axA.text(18.4, 12.6, r'$u(x,t)$, Benjamin$-$Ono', ha='center', fontsize=6)
axA.text(18.4, 9.6, 'space (vertical) $\\times$ time', ha='center', fontsize=5.5, color='#555555')
arrow(30.5, 30, 37.5, 30)

# A2 generator box
box(38.5, 6, 52, 47, FILL_K); axA.text(64.5, 55.5, r'Low-rank generator $\mathcal{G}_\theta$', ha='center', fontsize=7, fontweight='bold')
axA.text(64.5, 48.6, r'$\mathcal{G}_\theta u=\mathrm{Op}(\ell)u+\sum_r a_r(x)\,\mathrm{Op}(m_r)u+\mathrm{Op}(d)\,\Phi^{\mathrm{nl}}(u,\cdot)$',
         ha='center', fontsize=5.6)
mini = [(40.5, 27.5, r'$\ell(\xi)$', 'spectral'), (65.5, 27.5, r'$\sum_r a_r(x)\,m_r(\xi)$', 'position'),
        (40.5, 9.0, r'$d(\xi)$', 'flux'), (65.5, 9.0, r'$\Phi^{\mathrm{nl}}(u)$', 'nonlinear')]
for (x0, y0, ttl, _), kind in zip(mini, ('ell', 'sep', 'd', 'phi')):
    a = inset(x0 + 3, y0 + 2.5, 19, 11)
    if kind == 'ell':
        a.plot(xi_bo[band], ell_bo.imag[band], color=ORANGE, lw=1.3); a.set_xlabel(r'$\xi$', fontsize=5.5, labelpad=0.5)
    elif kind == 'sep':
        bd = xig <= 40; Z = Ksum.imag[:, bd]
        a.imshow(Z.T, origin='lower', aspect='auto', cmap='PuOr_r', vmin=-np.abs(Z).max(), vmax=np.abs(Z).max(), extent=(xg[0], xg[-1], 0, 40))
        a.set_xlabel(r'$x$', fontsize=5.5, labelpad=0.5); a.set_ylabel(r'$\xi$', fontsize=5.5, labelpad=0.5)
        for sp in a.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5)
    elif kind == 'd':
        a.plot(xi_bo[band], dsym.imag[band], color=ORANGE, lw=1.3, label='Im'); a.plot(xi_bo[band], dsym.real[band], color=GREY, lw=0.9, label='Re')
        a.set_xlabel(r'$\xi$', fontsize=5.5, labelpad=0.5); a.legend(fontsize=4.8, frameon=False, loc='upper left', handlelength=1.2)
    else:
        a.plot(ug.numpy(), phi_bo, color=ORANGE, lw=1.3); a.set_xlabel(r'$u$', fontsize=5.5, labelpad=0.5)
    a.set_title(ttl, fontsize=6, pad=1.5); a.set_xticks([]); a.set_yticks([])
arrow(91.5, 30, 98.5, 30)

# A3 exact time integration
axA.text(110.5, 55.5, 'Exact time integration', ha='center', fontsize=7, fontweight='bold')
a = inset(100.5, 15.6, 20, 24); heat(a, pred)
axA.text(110.5, 12.6, r'rollout, $e^{\Delta t\,\ell}$ exact', ha='center', fontsize=6)
axA.text(110.5, 9.6, '100 steps, unseen initial state', ha='center', fontsize=5.5, color='#555555')
arrow(122.5, 30, 129.5, 30)

# A4 automatic distillation → closed form
box(130.5, 6, 46, 47, FILL_Y); axA.text(153.5, 55.5, 'Automatic distillation', ha='center', fontsize=7, fontweight='bold')
P = ['0', '1/2', '1', '3/2', '2', '5/2', '3', '7/2', '4']
for j, p in enumerate(P):
    sel = (p == '2')
    axA.add_patch(Rectangle((133 + j * 4.6, 37.5), 4.0, 4.0, fc=ORANGE if sel else 'white', ec='#777777', lw=0.5))
    axA.text(135 + j * 4.6, 35.4, p, ha='center', va='top', fontsize=5)
axA.text(153.5, 44.6, r'dictionary $|\xi|^p$ (complex coefficients)', ha='center', fontsize=5.6)
axA.text(153.5, 31.0, 'sparse supports, held-out parsimony', ha='center', fontsize=5.6, color='#555555')
axA.annotate('', xy=(153.5, 22.5), xytext=(153.5, 29.0), arrowprops=dict(arrowstyle='-|>', lw=0.8, color=DARK, mutation_scale=7))
axA.text(153.5, 18.2, r'$\ell(\xi)=-1.0002\,i\,\xi|\xi|$', ha='center', fontsize=7, color=ORANGE)
axA.text(153.5, 12.6, r'$\approx-\mathcal{H}\,\partial_{xx}$  (truth: $-i\,\xi|\xi|$)', ha='center', fontsize=6.2)
axA.text(153.5, 8.8, 'closed-form dispersion relation', ha='center', fontsize=5.5, color='#555555')

# ================= B: dense vs low-rank symbol =================
label(0.005, 0.475, 'B')
x = np.linspace(-np.pi, np.pi, 128, endpoint=False); xi = np.arange(0, 41)
X, XI = np.meshgrid(x, xi, indexing='ij')
P2 = X * XI                                                  # Im p(x,xi) of G2 = x xi  (Re = -xi^2 is the spectral part)
y0, h = 0.10, 0.27
axk = fig.add_axes([0.035, y0, 0.15, h]); heat(axk, P2, 'PuOr_r')
axk.set_title(r'KANO: dense $p(x,\xi)$', fontsize=6.5, pad=2)
axk.set_xlabel(r'$x$', fontsize=6, labelpad=1); axk.set_ylabel(r'$\xi$', fontsize=6, labelpad=1)
fig.text(0.11, y0 - 0.055, r'$m^2$ symbol evaluations', ha='center', fontsize=6)
fig.text(0.205, y0 + h / 2, 'vs', ha='center', va='center', fontsize=7, color='#555555')
ax1 = fig.add_axes([0.245, y0 + 0.03, 0.035, h - 0.06]); ax1.plot(-xi ** 2, xi, color=ORANGE, lw=1.2); ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_title(r'$\ell(\xi)$', fontsize=6.5, pad=2)
fig.text(0.300, y0 + h / 2, '+', ha='center', va='center', fontsize=10)
ax2 = fig.add_axes([0.320, y0 + h - 0.045, 0.10, 0.035]); ax2.plot(x, x, color=BLUE, lw=1.2); ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_title(r'$a(x)=x$', fontsize=6, pad=1)
ax3 = fig.add_axes([0.320, y0, 0.10, h - 0.08]); heat(ax3, P2, 'PuOr_r'); ax3.set_xlabel(r'$a(x)\,m(\xi)$', fontsize=6, labelpad=1)
ax4 = fig.add_axes([0.427, y0, 0.035, h - 0.08]); ax4.plot(xi, xi, color=BLUE, lw=1.2); ax4.set_xticks([]); ax4.set_yticks([])
ax4.set_title(r'$m(\xi)=i\xi$', fontsize=6, pad=1)
fig.text(0.37, y0 - 0.055, r'KAFlow: $(R+1)\,m$ evaluations, $\mathcal{O}(Rm\log m)$', ha='center', fontsize=6)

# ================= C: singular values of the generator vs the propagator =================
label(0.525, 0.475, 'C')
axC = fig.add_axes([0.585, 0.09, 0.39, 0.33])
N = 128; xx = np.linspace(-np.pi, np.pi, N, endpoint=False); kk = np.fft.fftfreq(N, d=2 * np.pi / N) * 2 * np.pi
XX, KK = np.meshgrid(xx, kk, indexing='ij')
cases = [(r'$\mathcal{G}_1$ generator', XX ** 2 + KK ** 2, BLUE, '-'),
         (r'$\mathcal{G}_2$ generator', 1j * XX * KK - KK ** 2, SKY, '-'),
         ('Transport generator', 1j * 0.2 * np.sin(XX) * KK, PINK, '-')]
for eps, ls in ((0.1, ':'), (0.2, '--'), (0.3, '-.')):
    cases.append((rf'Propagator, $\varepsilon={eps}$', np.exp(1j * KK * eps * np.sin(XX)), ORANGE, ls))
for nm, S, col, ls in cases:
    s = np.linalg.svd(S.astype(np.complex128), compute_uv=False); s = np.clip(s / s[0], 1e-17, None)
    axC.plot(np.arange(1, 41), s[:40], ls, color=col, lw=1.2, label=nm)
axC.axhline(1e-4, color='#999999', lw=0.5, ls='-'); axC.text(40, 1.6e-4, r'$10^{-4}$', ha='right', fontsize=5.5, color='#777777')
axC.set_yscale('log'); axC.minorticks_off(); axC.set_xlim(1, 40); axC.set_ylim(1e-17, 2)
axC.set_xlabel('Singular value index $k$', fontsize=6.5); axC.set_ylabel(r'$\sigma_k/\sigma_1$ of the symbol', fontsize=6.5)
axC.legend(fontsize=5.5, loc='lower left', bbox_to_anchor=(0.1, 0.03), frameon=False, handlelength=2.4)
fig.savefig(f'{K}/paper/figs/fig1.pdf')
print('fig1 done', 'd(xi) imag/real max', np.abs(dsym.imag[band]).max(), np.abs(dsym.real[band]).max())
