"""High-rank synthetic operators (generality test). The input-function generator is that of the official KANO G_2 notebook (cell 4), unchanged:
 training and validation = Group A (sine_beats, chirped_cosine, periodic), test = Group B (wave_packet, sinc_pulse, gaussian_hermite), N=128, [-pi,pi).
 The true Gf is computed exactly in float64 with the same Kohn–Nirenberg quantization as KANO (full fftfreq, real part).
   H1  p = |k|^(1 + 0.5 sin x)                 fractional derivative whose order varies with position    separation rank (1e-4) 4
   H2  p = 1 / (1 + (1 + 0.5 sin x) k^2 / 25)  screened inverse with a position-dependent screening length  rank 4
   H3  p = exp(i k 0.2 sin x)  = f(x + 0.2 sin x)  coordinate warp                                       rank 16
 python -B bench/synthH_data.py
"""
import math, os
import numpy as np
import torch
from scipy.special import hermite

ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = ROOT


def outer_envelope(x):
    W = torch.zeros_like(x); li, lo = 5.0 * math.pi / 6.0, math.pi; a = x.abs()
    W[a <= li] = 1.0
    m = (a > li) & (a < lo); al = (a[m] - li) / (lo - li); W[m] = torch.cos(al * (math.pi / 2)) ** 4
    return W


def periodic(x):
    fr = np.random.uniform(0.5, 3.0) * 8.0; p1, p2 = np.random.uniform(0, 2 * math.pi, 2)
    return outer_envelope(x) * (torch.sin(fr * x + p1) + torch.cos(fr * x + p2))


def gaussian_hermite(x):
    deg = np.random.randint(1, 4); H = hermite(deg); mu = np.random.uniform(-2.0, 2.0); sg = np.random.uniform(0.5, 2.0) / 8.0
    poly = torch.tensor(H(((x - mu) / sg).numpy()), dtype=torch.float32)
    return outer_envelope(x) * poly * torch.exp(-((x - mu) ** 2) / (2 * sg ** 2))


def sine_beats(x):
    f1, f2 = np.random.uniform(0.5, 3.0, 2) * 8.0; p1, p2 = np.random.uniform(0, 2 * math.pi, 2)
    return outer_envelope(x) * torch.sin(f1 * x + p1) * torch.sin(f2 * x + p2)


def chirped_cosine(x):
    al = np.random.uniform(0.5, 2.0) * 12.0
    return outer_envelope(x) * torch.cos(al * x ** 2)


def wave_packet(x):
    mu = np.random.uniform(-2.0, 2.0); sg = np.random.uniform(0.5, 2.0) / 12.0; om = np.random.uniform(2.0, 6.0) * 12.0; ph = np.random.uniform(0, 2 * math.pi)
    return outer_envelope(x) * torch.exp(-((x - mu) ** 2) / (2 * sg ** 2)) * torch.sin(om * x + ph)


def sinc_pulse(x):
    al = np.random.uniform(0.5, 3.0) * 12.0; xa = al * x; b = torch.ones_like(x); m = xa.abs() > 1e-12; b[m] = torch.sin(xa[m]) / xa[m]
    return outer_envelope(x) * b


TRAIN = {'sine_beats': sine_beats, 'chirped_cosine': chirped_cosine, 'periodic': periodic}
UNSEEN = {'wave_packet': wave_packet, 'sinc_pulse': sinc_pulse, 'gaussian_hermite': gaussian_hermite}

SYMS = {'H1': lambda X, K: K.abs() ** (1 + 0.5 * torch.sin(X)) + 0j,
        'H2': lambda X, K: 1 / (1 + (1 + 0.5 * torch.sin(X)) * K ** 2 / 25) + 0j,
        'H3': lambda X, K: torch.exp(1j * K * 0.2 * torch.sin(X))}


def apply_kn(P, x64, f):                                  # Gf(x_n) = Re (dx dk / 2pi) sum_k sum_m p(x_n,k) e^{ik(x_n - x_m)} f_m
    N = x64.shape[0]; dx = 2 * math.pi / N
    k = torch.fft.fftfreq(N, d=dx, dtype=torch.float64) * 2 * math.pi
    E = torch.exp(-1j * k[:, None] * x64[None, :])        # (k,m)
    Bp = P * torch.exp(1j * k[None, :] * x64[:, None])     # (n,k)
    fac = dx * (k[1] - k[0]).item() / (2 * math.pi)
    return (fac * (Bp @ (E @ f.to(torch.complex128).T))).real.T


def make(op, fams, n_s, seed):
    np.random.seed(seed)
    x = torch.linspace(-math.pi, math.pi, 129, dtype=torch.float32)[:-1]
    x64 = torch.linspace(-math.pi, math.pi, 129, dtype=torch.float64)[:-1]
    k = torch.fft.fftfreq(128, d=2 * math.pi / 128, dtype=torch.float64) * 2 * math.pi
    P = SYMS[op](x64[:, None], k[None, :])
    names, Fs = [], []
    for _ in range(n_s):
        nm = np.random.choice(list(fams)); f = fams[nm](x); mv = f.abs().max().item()
        Fs.append(f / mv if mv > 1e-12 else f); names.append(nm)
    F = torch.stack(Fs).double(); G = apply_kn(P, x64, F).float()
    return [(x.clone(), F[i].float(), G[i], names[i]) for i in range(n_s)]


if __name__ == '__main__':
    # check: with the true G2 symbol, the same KN function must reproduce Gf of the official G_2 data
    d = torch.load(f'{R}/ext/KANO/G_2/dataset/test_2.pt', weights_only=False)
    x64 = torch.linspace(-math.pi, math.pi, 129, dtype=torch.float64)[:-1]
    k = torch.fft.fftfreq(128, d=2 * math.pi / 128, dtype=torch.float64) * 2 * math.pi
    F = torch.stack([e[1] for e in d]).double(); G = torch.stack([e[2] for e in d]).double()
    Gk = apply_kn(1j * x64[:, None] * k[None, :] - k[None, :] ** 2, x64, F)
    print('G2 reproduction relative error', float((Gk - G).norm() / G.norm()))
    for i, op in enumerate(('H1', 'H2', 'H3')):
        os.makedirs(f'{R}/data/synthH/{op}', exist_ok=True)
        for nm, fams, n_s, sd in (('train', TRAIN, 2000, 10 * i + 1), ('sample', TRAIN, 400, 10 * i + 2), ('test', UNSEEN, 400, 10 * i + 3)):
            ds = make(op, fams, n_s, sd)
            torch.save(ds, f'{R}/data/synthH/{op}/{nm}.pt')
            G = torch.stack([e[2] for e in ds])
            print(op, nm, len(ds), f'|Gf| rms {float(G.pow(2).mean().sqrt()):.3g}  min sample norm {float(G.norm(dim=1).min()):.3g}', flush=True)
