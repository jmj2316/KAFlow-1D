"""1D periodic PDE data generator (pseudo-spectral Lawson-IFRK4, 2/3 dealiasing).

Every equation is written as u_t = Lsym(xi) * u_hat  +  Nhat(u_hat).
The data are generated here because the exact equations must be known.
"""
import numpy as np
import os
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.abspath(__file__)))


def grid(N, L):
    k = np.arange(N // 2 + 1)
    xi = 2 * np.pi * k / L
    x = np.arange(N) * L / N
    return x, xi


EQS = {}


def _reg(name, L=2 * np.pi, dt_obs=0.02, doc=''):
    def deco(fn):
        EQS[name] = dict(build=fn, L=L, dt_obs=dt_obs, doc=doc)
        return fn
    return deco


@_reg('burgers', L=2 * np.pi, dt_obs=0.02, doc='u_t = nu u_xx - (u^2/2)_x, nu=0.02')
def burgers(xi):
    nu = 0.02
    Lsym = -nu * xi ** 2 + 0j
    d = -1j * xi
    return Lsym, d, lambda u: 0.5 * u ** 2


@_reg('kdv', L=2 * np.pi, dt_obs=0.01, doc='u_t = -u_xxx - 3 (u^2)_x')
def kdv(xi):
    Lsym = 1j * xi ** 3
    d = -3j * xi
    return Lsym, d, lambda u: u ** 2


@_reg('ks', L=22.0, dt_obs=0.2, doc='u_t = -u u_x - u_xx - u_xxxx')
def ks(xi):
    Lsym = xi ** 2 - xi ** 4 + 0j
    d = -1j * xi
    return Lsym, d, lambda u: 0.5 * u ** 2


@_reg('allen_cahn', L=2 * np.pi, dt_obs=0.05, doc='u_t = 0.01 u_xx + u - u^3')
def allen_cahn(xi):
    Lsym = -0.01 * xi ** 2 + 1.0 + 0j
    d = np.ones_like(xi) + 0j
    return Lsym, d, lambda u: -u ** 3


@_reg('fburgers', L=2 * np.pi, dt_obs=0.02, doc='u_t = -nu(-Delta)^{a/2}u - (u^2/2)_x, a=1.5, nu=0.05')
def fburgers(xi):
    nu, a = 0.05, 1.5
    Lsym = -nu * np.abs(xi) ** a + 0j
    d = -1j * xi
    return Lsym, d, lambda u: 0.5 * u ** 2


@_reg('bo', L=2 * np.pi, dt_obs=0.01, doc='Benjamin-Ono: u_t = -H u_xx - (u^2/2)_x  (symbol -i xi|xi|)')
def bo(xi):
    Lsym = -1j * xi * np.abs(xi)
    d = -1j * xi
    return Lsym, d, lambda u: 0.5 * u ** 2


@_reg('whitham', L=2 * np.pi, dt_obs=0.01, doc='u_t = -i xi sqrt(tanh(xi)/xi) u - (3/4)(u^2)_x')
def whitham(xi):
    c = np.ones_like(xi)
    nz = xi > 0
    c[nz] = np.sqrt(np.tanh(xi[nz]) / xi[nz])
    Lsym = -1j * xi * c
    d = -0.75j * xi * 2          # -(3/2) * i xi * (u^2/2)
    return Lsym, d, lambda u: 0.5 * u ** 2 * 1.0


def random_ic(rng, B, N, L, kmax=4, decay=1.0, std=1.0):
    x = np.arange(N) * L / N
    u = np.zeros((B, N))
    for k in range(1, kmax + 1):
        a = rng.standard_normal((B, 1)) * k ** (-decay)
        b = rng.standard_normal((B, 1)) * k ** (-decay)
        w = 2 * np.pi * k / L
        u += a * np.cos(w * x) + b * np.sin(w * x)
    u -= u.mean(1, keepdims=True)
    u *= std / (u.std(1, keepdims=True) + 1e-12)
    return u


def simulate(name, B, n_steps, N=128, seed=0, sub=20, kmax=4, std=1.0, dt_obs=None):
    cfg = EQS[name]
    L = cfg['L']
    dt_obs = dt_obs or cfg['dt_obs']
    x, xi = grid(N, L)
    Lsym, d, phi = cfg['build'](xi)
    mask = (np.arange(N // 2 + 1) <= N // 3).astype(float)   # 2/3 rule
    rng = np.random.default_rng(seed)
    u0 = random_ic(rng, B, N, L, kmax=kmax, std=std)
    dt = dt_obs / sub
    E = np.exp(Lsym * dt / 2)
    E2 = E * E

    def Nh(uh):
        u = np.fft.irfft(uh, n=N, axis=-1)
        return d * np.fft.rfft(phi(u), axis=-1) * mask

    uh = np.fft.rfft(u0, axis=-1) * mask
    out = np.zeros((B, n_steps + 1, N))
    out[:, 0] = np.fft.irfft(uh, n=N, axis=-1)
    for s in range(n_steps):
        for _ in range(sub):
            a = Nh(uh)
            b = Nh(E * (uh + 0.5 * dt * a))
            c = Nh(E * uh + 0.5 * dt * b)
            dd = Nh(E2 * uh + dt * E * c)
            uh = E2 * uh + dt / 6 * (E2 * a + 2 * E * (b + c) + dd)
        out[:, s + 1] = np.fft.irfft(uh, n=N, axis=-1)
    meta = dict(name=name, L=L, dt_obs=dt_obs, N=N)
    return out.astype(np.float32), meta


if __name__ == '__main__':
    import sys, os
    names = sys.argv[1:] or list(EQS)
    cfgs = dict(burgers=(256, 100), kdv=(256, 100), ks=(256, 100), allen_cahn=(256, 100),
                fburgers=(256, 100), bo=(256, 100), whitham=(256, 100))
    for nm in names:
        B, T = cfgs[nm]
        tr, meta = simulate(nm, B, T, seed=1)
        te, _ = simulate(nm, 64, T, seed=2)
        np.savez(f'{ROOT}/data/{nm}.npz', train=tr, test=te, **{k: v for k, v in meta.items()})
        print(nm, tr.shape, 'std', float(tr.std()), 'max|u|', float(np.abs(tr).max()), 'finite', bool(np.isfinite(tr).all()))
