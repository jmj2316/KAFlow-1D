"""Numbers for the tables and figures of the paper, computed from stored results and models only (no new training) → paper/data/numbers_G.json
 python -B paper/scripts/compute_numbers.py
"""
import sys, os, json, math, statistics as st
import numpy as np
import torch
import torch.nn as nn
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
K = ROOT; RK = f'{K}/ext/KANO'
sys.path.insert(0, K); sys.path.insert(0, f'{K}/bench')
torch.set_num_threads(8)
import ext_kano_official as KO
from kan import KAN
from neuralop.models import FNO
from neuralop.layers.embeddings import GridEmbeddingND
from kafu import KAFlowUOp2, spectrum_stats

ld = lambda p: torch.load(p, weights_only=False, map_location='cpu')
J = lambda p: json.load(open(p))
OUT = f'{K}/paper/data'; os.makedirs(OUT, exist_ok=True)
num = {}


def gdata(g, nm):
    d = ld(f'{RK}/G_{g}/dataset/{nm}_{g}.pt')
    return d[0][0].float(), torch.stack([e[1] for e in d]).float(), torch.stack([e[2] for e in d]).float(), [str(e[3]) for e in d]


def kano_official(g):
    if g == 1:
        sym = KAN(width=[2, 2], grid=10, k=3, seed=0, device='cpu', auto_save=False); act = nn.Identity()
    elif g == 2:
        sym = KAN(width=[[2, 0], [1, 1]], grid=10, k=3, seed=0, base_fun='identity', device='cpu', auto_save=False); act = nn.Identity()
    else:
        sym = KAN(width=[[2, 0], [1, 1]], grid=10, k=3, seed=10, base_fun='identity', device='cpu', auto_save=False)
        act = KAN(width=[2, 1], grid=10, k=3, seed=11, base_fun='identity', device='cpu', auto_save=False)
        act.load_state_dict(ld(f'{RK}/G_3/models/kanoactkan_3.pth'))
    sym.load_state_dict(ld(f'{RK}/G_{g}/models/kanosymkan_{g}.pth'))
    return KO.Kano(num_layers=1, symbol_net=sym, act_net=act).eval()


def kano_apply(model, x, F):                              # batched official KanoLayer (verified against the official output)
    lay = model.layers[0]; pdo = lay.pdo
    pdo.build_cache_for_x(x); N = x.shape[0]
    Xg = pdo.x_cache.view(N, 1).expand(N, N); Kg = pdo.k_cache.view(1, N).expand(N, N)
    a = pdo.a_net_2d(torch.stack([Xg.flatten(), Kg.flatten()], 1)); A = (a[:, 0] + 1j * a[:, 1]).view(N, N)
    out = []
    for i in range(0, F.shape[0], 50):
        t = torch.einsum('nyk,by->bnk', pdo.exp_factor_cache, F[i:i + 50].to(torch.complex64))
        gr = (pdo.factor * (A[None] * t).sum(-1)).real
        out.append(gr if isinstance(lay.act_net, nn.Identity) else lay.act_net(torch.stack([gr, F[i:i + 50]], -1).reshape(-1, 2)).reshape(gr.shape))
    return torch.cat(out)


def fno_official(g):
    pe = GridEmbeddingND(in_channels=1, dim=1, grid_boundaries=[[0, 1]])
    m = FNO(n_modes=(128,), hidden_channels=64, in_channels=1, out_channels=1, positional_embedding=pe, n_layers=2,
            fno_skip='linear', channel_mlp_skip='soft-gating')
    m.load_state_dict({k: v for k, v in ld(f'{RK}/G_{g}/models/fno_{g}.pth').items() if k != '_metadata'}, strict=False)
    return m.eval()


def kaflow(g, s, tag='U2opd'):
    _, Ftr, Gtr, _ = gdata(g, 'train')
    E, xr = spectrum_stats(Ftr, 2 * math.pi); sc = float(Gtr.pow(2).mean().sqrt() / Ftr.pow(2).mean().sqrt())
    m = KAFlowUOp2(128, 2 * math.pi, E, xr, sc, R=2, J=1, H=2); m.load_state_dict(ld(f'{K}/runs/synth/G{g}_{tag}_s{s}.pt'))
    return m.eval()


rel = lambda P, G: ((P - G).norm(dim=1) / G.norm(dim=1))

# ---------------- 1) G1–G3 table (KANO Table 1 format: A=training, B=Group B test)
num['G'] = {}
for g in (1, 2, 3):
    r = [J(f'{K}/runs/synth/G{g}_U2opd_s{s}.json') for s in range(6)]
    row = dict(A=st.mean(d['train'] for d in r), B=st.mean(d['test'] for d in r), B_std=st.stdev(d['test'] for d in r), params=r[0]['params'])
    if g < 3:
        sy = [J(f'{K}/runs/synth/G{g}_U2sym_s{s}.json') for s in range(6)]
        sym = dict(A=st.mean(d['train'] for d in sy), B=st.mean(d['test'] for d in sy), coef=[d['coef'] for d in sy],
                   maxerr=max(d['max_abs_coef_err'] for d in sy), support=[d['support'] for d in sy], phi=[d['use_phi'] for d in sy])
    else:                                                                     # G3 fully symbolic form: rebuild the closed form from the coefficients and compute the training-set loss
        sy = [J(f'{K}/runs/synth/G3_U2symnl_s{s}.json') for s in range(6)]
        x, Ftr, Gtr, _ = gdata(3, 'train'); _, Fte, Gte, _ = gdata(3, 'test')
        def closed(c, F):
            F = F.double(); xx = x.double(); fh = torch.fft.rfft(F, dim=-1)
            xi = torch.fft.rfftfreq(128, d=2 * math.pi / 128).double() * 2 * math.pi
            D = lambda p: torch.fft.irfft((1j * xi) ** p * fh, n=128, dim=-1) if p else F
            out = torch.zeros_like(F)
            for key, v in c.items():
                if key.startswith('x^'):
                    n, p = int(key[2]), int(key.split('d^')[1]); out += v * xx ** n * D(p)
                else:
                    fx, fxx = D(1), D(2)
                    out += v * {'f^2': F ** 2, 'f^3': F ** 3, 'f f_x': F * fx, 'f^2 f_x': F ** 2 * fx, 'f f_xx': F * fxx, 'f_x^2': fx ** 2,
                                'x f^2': xx * F ** 2, 'x f^3': xx * F ** 3}[key]
            return out
        trs = [float(rel(closed(d['coef'], Ftr), Gtr.double()).mean()) for d in sy]
        tes = [float(rel(closed(d['coef'], Fte), Gte.double()).mean()) for d in sy]
        sym = dict(A=st.mean(trs), B=st.mean(tes), B_json=st.mean(d['test'] for d in sy), coef=[d['coef'] for d in sy],
                   maxerr=max(d['max_abs_coef_err'] for d in sy), nl=[d['nl_support'] for d in sy])
    num['G'][g] = dict(kaflow=row, kaflow_symbolic=sym)
    print('G', g, row, {k: v for k, v in sym.items() if k in ('A', 'B', 'B_json', 'maxerr')}, flush=True)

# ---------------- 2) losses by function family (Group A held-out sample / Group B test), 3) interpolation test (true operator applied)
TRUEG = {1: lambda f, fx, fxx, x: x ** 2 * f - fxx, 2: lambda f, fx, fxx, x: x * fx + fxx, 3: lambda f, fx, fxx, x: f ** 3 + x * fx + fxx}


def true_op(g, x, F):                                    # same spectral derivatives as the KANO generator
    n = F.shape[-1]; dx = (x[-1] - x[0] + (x[1] - x[0])) / n
    k = torch.fft.fftfreq(n, d=float(dx)) * (2 * math.pi); fh = torch.fft.fft(F, dim=-1)
    fx = torch.fft.ifft(1j * k * fh, dim=-1).real; fxx = torch.fft.ifft(-(k ** 2) * fh, dim=-1).real
    return TRUEG[g](F, fx, fxx, x)


num['family'] = {}; num['interp'] = {}
for g in (1, 2, 3):
    x, Fa, Ga, na = gdata(g, 'sample'); _, Fb, Gb, nb = gdata(g, 'test')
    kano, fno = kano_official(g), fno_official(g); kafs = [kaflow(g, s) for s in range(6)]
    with torch.no_grad():
        preds = {'KAFlow': lambda F: torch.stack([m(F) for m in kafs]), 'KANO': lambda F: kano_apply(kano, x, F)[None],
                 'FNO': lambda F: fno(F.unsqueeze(1)).squeeze(1)[None]}
        fam = {}
        for nm, fn in preds.items():
            ra = torch.stack([rel(P, Ga) for P in fn(Fa)]).mean(0); rb = torch.stack([rel(P, Gb) for P in fn(Fb)]).mean(0)
            d = {}
            for n_, v in zip(na, ra.tolist()):
                d.setdefault(n_, []).append(v)
            for n_, v in zip(nb, rb.tolist()):
                d.setdefault(n_, []).append(v)
            fam[nm] = {k: sum(v) / len(v) for k, v in d.items()}
        num['family'][g] = fam
        al = np.linspace(0, 1, 101); cur = {nm: [] for nm in preds}
        for a in al:
            Fm = (1 - a) * Fa + a * Fb; Gm = true_op(g, x, Fm)
            for nm, fn in preds.items():
                cur[nm].append(float(torch.stack([rel(P, Gm) for P in fn(Fm)]).mean()))
        num['interp'][g] = {nm: [v / c[0] for v in c] for nm, c in cur.items()}
        num['interp'][g]['abs'] = cur
    print('family/interp', g, {nm: round(num['interp'][g][nm][-1], 3) for nm in preds}, flush=True)

# ---------------- 4) slices of the learned symbols (G1–G3) and Phi of G3
num['slices'] = {}
for g in (1, 2, 3):
    m = kaflow(g, 0)
    with torch.no_grad():
        xg = m.x.double(); xi = m.xi.double()
        ell = m.ell()[0].to(torch.complex128)
        Kx = (m.ar_scale * m.ar().double()[:, :, None] * m.mr().to(torch.complex128)[:, None, :]).sum(0)
        P = ell[None, :] + Kx                                                # p(x, xi), xi >= 0 (half rfft grid)
        i0 = int(torch.argmin(xg.abs()))
        d = dict(x=xg.tolist(), xi=xi.tolist(), p_xi0_re=P[:, 0].real.tolist(), p_x0_re=P[i0].real.tolist(), p_x0_im=P[i0].imag.tolist())
        j = int(torch.argmin((xi - 10).abs())); d['xi_j'] = float(xi[j]); d['p_xij_im'] = (P[:, j].imag / xi[j]).tolist()
        if g == 3:
            _, Fte, _, _ = gdata(3, 'test'); fh = torch.fft.rfft(Fte, dim=-1)
            feats = torch.cat([Fte.unsqueeze(-1), torch.fft.irfft(m.M()[None] * fh[:, None, :], n=128, dim=-1).transpose(1, 2)], -1)
            ph = m.phi(feats, m.x)
            sel = torch.randperm(Fte.numel())[:3000]
            d['phi_f'] = Fte.flatten()[sel].tolist(); d['phi_val'] = ph.flatten()[sel].tolist()
        num['slices'][g] = d

json.dump(num, open(f'{OUT}/numbers_G.json', 'w'))
print('saved', flush=True)
