"""Reproduction of the interpolation test of the KANO paper + the KANO_symbolic row.
 Official procedure (G_1 notebook cells 52-54): ds_test=sample (Group A held out, 400) and ds_unseen=test (Group B, 400) are paired by index,
 f=(1-a)f1+a f2, Gf=(1-a)Gf1+a Gf2, a = linspace(0,1,101). Metric = mean relative L2, Loss Ratio = L(a)/L(0).
 Models: KANO (official weights), KANO_symbolic (official weights), FNO (official weights, verified setting), KAFlow-U (seeds 0..5).
 python -B interp_test.py 1
"""
import sys, os, json, math
import numpy as np
import torch
import torch.nn as nn
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'bench'))
import ext_kano_official as KO
from kan import KAN
from neuralop.models import FNO
from neuralop.layers.embeddings import GridEmbeddingND
from kafu import KAFlowUOp, KAFlowUOp2, spectrum_stats

torch.set_num_threads(8)
R = ROOT; RK = f'{R}/ext/KANO'
g = int(sys.argv[1]); dev = torch.device('cpu')
ld = lambda p: torch.load(p, weights_only=False, map_location='cpu')


class SymCorrect(nn.Module):                       # official symbol_correct (G_1: quadratic in x,k / G_2,G_3: Re=quadratic(k), Im=(a0+a1x)(c0+c1k))
    def __init__(self, g):
        super().__init__(); self.g = g
        if g == 1:
            self.a = nn.Parameter(torch.zeros(3)); self.b = nn.Parameter(torch.zeros(3))
        else:
            self.b = nn.Parameter(torch.zeros(3)); self.a = nn.Parameter(torch.zeros(2)); self.c = nn.Parameter(torch.zeros(2))

    def forward(self, xk):
        x, k = xk[..., 0], xk[..., 1]
        if self.g == 1:
            real = (self.a[0] + self.a[1] * x + self.a[2] * x ** 2) + (self.b[0] + self.b[1] * k + self.b[2] * k ** 2)
            return torch.stack([real, torch.zeros_like(real)], -1)
        real = self.b[0] + self.b[1] * k + self.b[2] * k ** 2
        imag = (self.a[0] + self.a[1] * x) * (self.c[0] + self.c[1] * k)
        return torch.stack([real, imag], -1)


class ActCorrect(nn.Module):                       # official act_correct (G_3): (l0+l1 g) + (c0+c1 f+c2 f^2+c3 f^3)
    def __init__(self):
        super().__init__(); self.lin_params = nn.Parameter(torch.zeros(2)); self.cub_params = nn.Parameter(torch.zeros(4))

    def forward(self, x):
        gr, f = x[..., 0], x[..., 1]; l0, l1 = self.lin_params; c0, c1, c2, c3 = self.cub_params
        return (l0 + l1 * gr + c0 + c1 * f + c2 * f ** 2 + c3 * f ** 3).unsqueeze(-1)


def kano_numeric():
    if g == 1:
        sym = KAN(width=[2, 2], grid=10, k=3, seed=0, device=dev); act = nn.Identity()
    elif g == 2:
        sym = KAN(width=[[2, 0], [1, 1]], grid=10, k=3, seed=0, base_fun='identity', device=dev); act = nn.Identity()
    else:
        sym = KAN(width=[[2, 0], [1, 1]], grid=10, k=3, seed=10, base_fun='identity', device=dev)
        act = KAN(width=[2, 1], grid=10, k=3, seed=11, base_fun='identity', device=dev)
        act.load_state_dict(ld(f'{RK}/G_3/models/kanoactkan_3.pth'))
    sym.load_state_dict(ld(f'{RK}/G_{g}/models/kanosymkan_{g}.pth'))
    return KO.Kano(num_layers=1, symbol_net=sym, act_net=act).eval()


def kano_symbolic():
    sym = SymCorrect(g); sym.load_state_dict(ld(f'{RK}/G_{g}/models/kanosymans_{g}.pth'))
    act = nn.Identity()
    if g == 3:
        act = ActCorrect(); act.load_state_dict(ld(f'{RK}/G_3/models/kanoactans_3.pth'))
    return KO.Kano(num_layers=1, symbol_net=sym, act_net=act).eval()


def kano_batched(model, x, F):
    """Batched official KanoLayer: the double sum of pdo is unchanged, only the sample dimension is vectorized."""
    lay = model.layers[0]; pdo = lay.pdo
    pdo.build_cache_for_x(x); N = x.shape[0]
    Xg = pdo.x_cache.view(N, 1).expand(N, N); Kg = pdo.k_cache.view(1, N).expand(N, N)
    a = pdo.a_net_2d(torch.stack([Xg.flatten(), Kg.flatten()], 1))
    A = (a[:, 0] + 1j * a[:, 1]).view(N, N)
    out = []
    for i in range(0, F.shape[0], 50):
        t = torch.einsum('nyk,by->bnk', pdo.exp_factor_cache, F[i:i + 50].to(torch.complex64))
        gr = (pdo.factor * (A[None] * t).sum(-1)).real
        if isinstance(lay.act_net, nn.Identity):
            out.append(gr)
        else:
            out.append(lay.act_net(torch.stack([gr, F[i:i + 50]], -1).reshape(-1, 2)).reshape(gr.shape))
    return torch.cat(out)


def fno():
    pe = GridEmbeddingND(in_channels=1, dim=1, grid_boundaries=[[0, 1]])     # setting that reproduces Table 1 (verified)
    m = FNO(n_modes=(128,), hidden_channels=64, in_channels=1, out_channels=1, positional_embedding=pe, n_layers=2,
            fno_skip='linear', channel_mlp_skip='soft-gating')
    sd = {k: v for k, v in ld(f'{RK}/G_{g}/models/fno_{g}.pth').items() if k != '_metadata'}
    m.load_state_dict(sd, strict=False)
    return m.eval()


def data(nm):
    d = torch.load(f'{RK}/G_{g}/dataset/{nm}_{g}.pt', weights_only=False, map_location='cpu')
    return d[0][0].float(), torch.stack([e[1] for e in d]).float(), torch.stack([e[2] for e in d]).float()


x, FA, GA = data('sample'); _, FB, GB = data('test'); _, Ftr, Gtr = data('train')
rel = lambda p, t: float(((p - t).norm(dim=1) / t.norm(dim=1)).mean())

models = {}
with torch.no_grad():
    kn = kano_numeric(); ks = kano_symbolic()
    for nm, m in (('KANO', kn), ('KANO_symbolic', ks)):                        # batched computation = check against the official per-sample computation
        ref = torch.stack([m(x, FB[i]) for i in range(3)]); mine = kano_batched(m, x, FB[:3])
        assert float((ref - mine).abs().max() / ref.abs().max()) < 1e-5, nm
    models['KANO'] = lambda F: kano_batched(kn, x, F)
    models['KANO_symbolic'] = lambda F: kano_batched(ks, x, F)
    fm = fno(); models['FNO'] = lambda F, fm=fm: fm(F.unsqueeze(1)).squeeze(1)
    E, xi_ref = spectrum_stats(Ftr, 2 * math.pi)
    scale = float(Gtr.pow(2).mean().sqrt() / Ftr.pow(2).mean().sqrt())
    for s in range(6):
        p = f'{R}/runs/synth/G{g}_U_s{s}.pt'
        if not os.path.exists(p):
            continue
        m = KAFlowUOp(128, 2 * math.pi, E, xi_ref, scale, J=1, H=2); m.load_state_dict(ld(p)); m.eval()
        models[f'KAFlow-U_s{s}'] = lambda F, m=m: m(F)
    for s in range(6):                                                         # v2: position-dependent low-rank symbol
        p = f'{R}/runs/synth/G{g}_U2op_s{s}.pt'
        if not os.path.exists(p):
            continue
        m = KAFlowUOp2(128, 2 * math.pi, E, xi_ref, scale, R=2, J=1, H=2); m.load_state_dict(ld(p)); m.eval()
        models[f'KAFlow-U2_s{s}'] = lambda F, m=m: m(F)
        if s == 0:                                                             # symbolic recovery: effective linear symbol = ell(xi) + sum_r scale a_r(x) m_r(xi)
            xx = m.x.double(); xi = m.xi.double()
            Ksym = m.ell()[0].double()[None, :] + (m.ar_scale * m.ar().double()[:, :, None] * m.mr().to(torch.complex128)[:, None, :]).sum(0)
            ik = 1j * xi
            basis, names = [], []
            for n in range(4):
                for pp in range(3):
                    basis.append((xx[:, None] ** n) * (ik[None, :] ** pp)); names.append(f'x^{n}(i xi)^{pp}')
            band = (xi <= xi[-1] * 0.5)                                       # mainly the data band (unidentifiable high wavenumbers excluded)
            A = torch.stack([b[:, band].reshape(-1) for b in basis], 1)
            c = torch.linalg.lstsq(A, Ksym[:, band].reshape(-1, 1)).solution[:, 0]
            symb = {nm_: [round(float(v.real), 4), round(float(v.imag), 4)] for nm_, v in zip(names, c) if abs(v) > 5e-3}

    # Table 1 columns (train/test) with the same harness: check of the KANO_symbolic row
    table = {nm: dict(test=rel(f(FB), GB)) for nm, f in models.items()}
    alphas = np.linspace(0.0, 1.0, 101); curves = {nm: [] for nm in models}
    for a in alphas:
        Fi = (1 - a) * FA + a * FB; Gi = (1 - a) * GA + a * GB
        for nm, f in models.items():
            curves[nm].append(rel(f(Fi), Gi))

out = dict(g=g, alphas=alphas.tolist(), curves=curves, table=table, symbolic_v2=symb)
os.makedirs(f'{R}/runs/interp', exist_ok=True)
json.dump(out, open(f'{R}/runs/interp/G{g}.json', 'w'))
for nm, c in curves.items():
    c = np.array(c)
    print(f'G{g} {nm:16s} test(B) {table[nm]["test"]:.3e} | L(0) {c[0]:.3e} L(0.5) {c[50]:.3e} L(1) {c[-1]:.3e} | max Loss Ratio {np.max(c / c[0]):.2f}', flush=True)
