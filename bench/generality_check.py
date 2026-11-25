"""Generality check: (A) separation rank of the symbol: true operator / symbol learned by the official KANO / upper bound of the KAFlow-U construction
               (B) speed of an O(N^2) implementation of KANO (FFT followed by a dense KN sum), a fair comparison with the official O(N^3) algorithm in speed.py
 python -B bench/generality_check.py
"""
import sys, math, json, time
import numpy as np
import torch
import torch.nn as nn
import os
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'bench'))
torch.set_num_threads(8)
import ext_kano_official as KO
from kan import KAN
from kafu import KAFlowUOp2

R = ROOT; RK = f'{R}/ext/KANO'
ld = lambda p: torch.load(p, weights_only=False, map_location='cpu')


def build(g):                                             # same as kano_verify.py (official weights)
    if g == 1:
        sym = KAN(width=[2, 2], grid=10, k=3, seed=0, device='cpu'); act = nn.Identity()
    elif g == 2:
        sym = KAN(width=[[2, 0], [1, 1]], grid=10, k=3, seed=0, base_fun='identity', device='cpu'); act = nn.Identity()
    else:
        sym = KAN(width=[[2, 0], [1, 1]], grid=10, k=3, seed=10, base_fun='identity', device='cpu')
        act = KAN(width=[2, 1], grid=10, k=3, seed=11, base_fun='identity', device='cpu')
        act.load_state_dict(ld(f'{RK}/G_3/models/kanoactkan_3.pth'))
    sym.load_state_dict(ld(f'{RK}/G_{g}/models/kanosymkan_{g}.pth'))
    return KO.Kano(num_layers=1, symbol_net=sym, act_net=act).eval()


def symbol(model, x):
    pdo = model.layers[0].pdo; pdo.build_cache_for_x(x); N = x.shape[0]
    Xg = pdo.x_cache.view(N, 1).expand(N, N); Kg = pdo.k_cache.view(1, N).expand(N, N)
    a = pdo.a_net_2d(torch.stack([Xg.flatten(), Kg.flatten()], 1))
    return (a[:, 0] + 1j * a[:, 1]).view(N, N), pdo


def kano_n3(model, x, F):                                 # batched as in speed.py (official algorithm, O(N^3) per sample)
    A, pdo = symbol(model, x)
    t = torch.einsum('nyk,by->bnk', pdo.exp_factor_cache, F.to(torch.complex64))
    return (pdo.factor * (A[None] * t).sum(-1)).real


def kano_B(model, x):
    """Dense KN: gr = Re(Bp @ E @ F), Bp[n,k] = factor p(x_n,k) e^{ik x_n}, E[k,m] = e^{-ik x_m}.
    The float32 data grid is not exactly uniform (error 1.3e-5), so a DFT on the actual x_m replaces the FFT, which gives the same operator as the official code."""
    A, pdo = symbol(model, x)
    Bp = pdo.factor * A * torch.exp(1j * pdo.k_cache[None, :] * x[:, None])
    E = torch.exp(-1j * pdo.k_cache[:, None] * x[None, :])
    return Bp, E


def kano_n2(model, x, F, W=None):                         # O(N^2) per sample; with W (real N×N inference cache) the symbol evaluation is skipped
    if W is None:
        Bp, E = kano_B(model, x)
        gr = ((F.to(torch.complex64) @ E.T) @ Bp.T).real
    else:
        gr = F @ W.T
    lay = model.layers[0]
    return gr if isinstance(lay.act_net, nn.Identity) else lay.act_net(torch.stack([gr, F], -1).reshape(-1, 2)).reshape(gr.shape)


def timeit(fn, reps=5):
    fn(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def sv_report(M):
    s = torch.linalg.svdvals(M.to(torch.complex128)); s = (s / s[0]).numpy()
    return dict(rank_1e3=int((s > 1e-3).sum()), rank_1e6=int((s > 1e-6).sum()), sv=[float(f'{v:.2e}') for v in s[:6]])


res = dict(rank={}, speed={}, check={})
# ---------- (A) separation rank
N = 128; x = -math.pi + torch.arange(N, dtype=torch.float32) * 2 * math.pi / N
k = torch.fft.fftfreq(N, d=2 * math.pi / N) * 2 * math.pi
X, K = x.double()[:, None], k.double()[None, :]
true = {1: X ** 2 + K ** 2, 2: 1j * X * K - K ** 2, 3: 1j * X * K - K ** 2}
with torch.no_grad():
    for g in (1, 2, 3):
        A, _ = symbol(build(g), x)
        res['rank'][f'G{g}'] = dict(true=sv_report(true[g] + 0j), kano_learned=sv_report(A))
        print(f'G{g}: true {res["rank"][f"G{g}"]["true"]} | KANO learned {res["rank"][f"G{g}"]["kano_learned"]}', flush=True)
    # auxiliary: typical high-rank symbols (fractional derivative whose order varies with position, position-dependent screening length)
    for nm, P in (('|xi|^(1+0.5 sin x)', (K.abs() + 1e-12) ** (1 + 0.5 * torch.sin(X))),
                  ('exp(-(1+0.5 sin x) xi^2/50)', torch.exp(-(1 + 0.5 * torch.sin(X)) * K ** 2 / 50))):
        res['rank'][nm] = dict(true=sv_report(P + 0j)); print(nm, res['rank'][nm], flush=True)

# ---------- (B) speed: optimized O(N^2) KANO
kano = build(2)
d = ld(f'{RK}/G_2/dataset/test_2.pt'); x128 = d[0][0].float(); F = torch.stack([e[1] for e in d]).float()
with torch.no_grad():
    a, b = kano_n3(kano, x128, F[:50]), kano_n2(kano, x128, F[:50])
    res['check']['n2_vs_n3_rel'] = float((a - b).norm() / a.norm())
    G = torch.stack([e[2] for e in d]).float(); rl = lambda p: float(((p - G).norm(dim=1) / G.norm(dim=1)).mean())
    res['check']['test_relL2_n3'] = rl(torch.cat([kano_n3(kano, x128, F[i:i + 50]) for i in range(0, 400, 50)]))
    res['check']['test_relL2_n2'] = rl(kano_n2(kano, x128, F)); print('O(N^2) vs official algorithm, relative difference', res['check'], flush=True)
    Bp, E = kano_B(kano, x128); B = (Bp @ E).real
    res['speed']['infer_400_KANO_n2_symbol_eval'] = timeit(lambda: kano_n2(kano, x128, F))
    res['speed']['infer_400_KANO_n2_cached'] = timeit(lambda: kano_n2(kano, x128, F, B))
dtr = ld(f'{RK}/G_2/dataset/train_2.pt'); Ftr = torch.stack([e[1] for e in dtr]).float(); Gtr = torch.stack([e[2] for e in dtr]).float()


def step():
    bi = torch.randint(0, Ftr.shape[0], (64,)); out = kano_n2(kano, x128, Ftr[bi])
    loss = ((out - Gtr[bi]).norm(dim=1) / Gtr[bi].norm(dim=1)).mean()
    for p in kano.parameters():
        p.grad = None
    loss.backward()
res['speed']['train_step_64_KANO_n2'] = timeit(step)
print({k_: round(v * 1e3, 2) for k_, v in res['speed'].items()}, '(ms)', flush=True)

res['scaling'] = {}
for N in (64, 128, 256, 512, 1024):
    xs = -math.pi + torch.arange(N, dtype=torch.float32) * 2 * math.pi / N; fN = torch.randn(16, N)
    EN = torch.ones(N // 2 + 1); EN[0] = 1e-3
    kafN = KAFlowUOp2(N, 2 * math.pi, EN, 3.0, 100.0).eval()
    with torch.no_grad():
        Bp, E = kano_B(kano, xs); BN = (Bp @ E).real
        row = {'KANO O(N^3) official algorithm': timeit(lambda: kano_n3(kano, xs, fN), reps=3) if N <= 512 else None,
               'KANO O(N^2) incl. symbol evaluation': timeit(lambda: kano_n2(kano, xs, fN), reps=3),
               'KANO O(N^2) cached (inference)': timeit(lambda: kano_n2(kano, xs, fN, BN)),
               'KAFlow-U': timeit(lambda: kafN(fN))}
    res['scaling'][N] = row
    print(f'N={N}: ' + ' | '.join(f'{k_} {"-" if v is None else f"{v*1e3:.2f}ms"}' for k_, v in row.items()), flush=True)
json.dump(res, open(f'{R}/runs/generality_check.json', 'w'), ensure_ascii=False, indent=1)
