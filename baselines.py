"""Baselines: FNO1d (autoregressive one-step), KANO-lite (dual-domain pseudo-differential layer, reimplementation).
python -B baselines.py fno burgers   |   python -B baselines.py kano burgers
"""
import sys, time, json, math
import numpy as np
import torch
import torch.nn as nn
from kaf import KANLayer
import os
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.abspath(__file__)))

torch.set_num_threads(8)


# ---------------- FNO1d ----------------
class SpecConv(nn.Module):
    """Channels-last (B,N,C) layout: rfft along N. Implemented without conv/transpose (avoids the nondeterministic CPU backward pass)."""

    def __init__(self, c, modes):
        super().__init__()
        self.modes = modes
        self.w = nn.Parameter((1.0 / (c * c)) * torch.rand(c, c, modes, dtype=torch.cfloat))   # standard FNO initialization

    def forward(self, h):                       # (B,N,C)
        N = h.shape[1]
        xh = torch.fft.rfft(h, dim=1)           # (B,K,C)
        y = torch.einsum('bmi,iom->bmo', xh[:, :self.modes], self.w)
        pad = torch.zeros(y.shape[0], xh.shape[1] - self.modes, y.shape[2], dtype=y.dtype, device=y.device)
        return torch.fft.irfft(torch.cat([y, pad], 1), n=N, dim=1)


class FNO1d(nn.Module):
    def __init__(self, width=32, modes=16, layers=4):
        super().__init__()
        self.lift = nn.Linear(2, width)
        self.sp = nn.ModuleList([SpecConv(width, modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, u):                       # (B,N) -> next (B,N)  (residual form)
        B, N = u.shape
        x = torch.linspace(0, 1, N, device=u.device).expand(B, N)
        h = self.lift(torch.stack([u, x], -1))
        for s, p in zip(self.sp, self.pw):
            h = torch.nn.functional.gelu(s(h) + p(h))
        return u + self.proj(h).squeeze(-1)


# ---------------- KANO-lite ----------------
class KANOLayer(nn.Module):
    """L(a)(x) = Phi( F^-1[ p(x,xi) F a ](x), a(x) ),  p = KAN(x, xi) -> (Re, Im)  (Kohn-Nirenberg double sum)"""

    def __init__(self, N, L, hid=5, G=10):
        super().__init__()
        self.N = N
        k = torch.arange(N // 2 + 1, dtype=torch.float32)
        xi = 2 * math.pi * k / L
        x = torch.arange(N, dtype=torch.float32) * L / N
        self.register_buffer('xi', xi); self.register_buffer('x', x)
        self.register_buffer('xin', xi / xi[-1] * 2 - 1)
        self.register_buffer('xn', x / L * 2 - 1)
        self.p1 = KANLayer(2, hid, G=G, lo=-1.0, hi=1.0)
        self.p2 = KANLayer(hid, 2, G=G, lo=-3.0, hi=3.0)
        self.f1 = KANLayer(2, hid, G=G, lo=-4.0, hi=4.0)
        self.f2 = KANLayer(hid, 1, G=G, lo=-4.0, hi=4.0)
        self.log_amp = nn.Parameter(torch.zeros(()))
        self.L = L
        phase = x[:, None, None] * 0
        self.register_buffer('ph', x[:, None] - x[None, :])             # (N,N)

    def symbol(self):
        X, XI = torch.meshgrid(self.xn, self.xin, indexing='ij')       # (N,K)
        z = torch.stack([X, XI], -1)
        out = self.p2(torch.tanh(self.p1(z))) * self.log_amp.exp()
        return torch.complex(out[..., 0], out[..., 1])                 # (N,K)

    def forward(self, a):                                              # (B,N)
        p = self.symbol()                                              # (N,K)
        ah = torch.fft.rfft(a, dim=-1)
        # (F^-1[p(x,.) ah])(x) = sum_k w_k Re/handle real: use irfft-like weights
        w = torch.full_like(self.xi, 2.0); w[0] = 1.0
        if self.N % 2 == 0:
            w[-1] = 1.0
        ex = torch.exp(1j * self.x[:, None] * self.xi[None, :])         # (N,K)
        y = torch.einsum('xk,xk,bk->bx', ex, p * w, ah).real / self.N
        f = torch.stack([y, a], -1)
        return self.f2(torch.tanh(self.f1(f))).squeeze(-1)


class KANOLite(nn.Module):
    def __init__(self, N, L, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([KANOLayer(N, L) for _ in range(layers)])

    def forward(self, u):
        h = u
        for l in self.layers:
            h = l(h)
        return u + h


def clip_grad(params, max_norm):
    """Gradient-norm clipping that supports complex parameters (clip_grad_norm_ of torch 2.1 returns inf for complex tensors)"""
    ps = [p for p in params if p.grad is not None]
    tot = torch.sqrt(sum((p.grad.abs() ** 2).sum() for p in ps))
    coef = min(1.0, max_norm / (float(tot) + 1e-6))
    for p in ps:
        p.grad.mul_(coef)
    return tot


def run(kind, eq, iters=3000, bs=32, seed=0):
    torch.manual_seed(seed)
    D = np.load(f'{ROOT}/data/{eq}.npz')
    tr = torch.from_numpy(D['train']); te = torch.from_numpy(D['test'])
    N, L = int(D['N']), float(D['L'])
    model = FNO1d() if kind == 'fno' else KANOLite(N, L)
    npar = sum(p.numel() for p in model.parameters())
    print(kind, eq, 'params', npar, flush=True)
    # zero-initialize the last layer so that the residual map starts at the identity (prevents rollout blow-up)
    last = model.proj[-1] if kind == 'fno' else None
    if last is not None:
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = [(1, iters // 3), (4, iters // 3), (16, iters - 2 * (iters // 3))]   # same curriculum as KAFlow
    total = iters; it = 0
    t0 = time.time()
    for K, n_it in sched:
        for _ in range(n_it):
            lr = 1e-3 * 0.5 * (1 + math.cos(math.pi * it / total)) + 1e-5
            for g in opt.param_groups:
                g['lr'] = lr
            b = torch.randint(0, tr.shape[0], (bs,)); s0 = torch.randint(0, tr.shape[1] - K, (bs,))
            seg = tr[b[:, None], s0[:, None] + torch.arange(K + 1)[None]]
            h = seg[:, 0]; loss = 0
            for k in range(1, K + 1):
                h = model(h)
                loss = loss + ((h - seg[:, k]).flatten(1).norm(dim=1) / seg[:, k].flatten(1).norm(dim=1)).mean()
            loss = loss / K
            opt.zero_grad(); loss.backward(); clip_grad(model.parameters(), 1.0); opt.step()
            it += 1
            if it % 200 == 0:
                print(f'it {it} K {K} loss {loss.item():.4e} t {time.time()-t0:.0f}s', flush=True)
    model.eval()
    with torch.no_grad():
        h = te[:, 0]; outs = [h]
        for _ in range(te.shape[1] - 1):
            h = model(h); outs.append(h)
        pred = torch.stack(outs, 1)
    err = (pred - te).flatten(2).norm(dim=2) / te.flatten(2).norm(dim=2)
    rmse = ((pred - te) ** 2).mean((0, 2)).sqrt()
    res = dict(eq=eq, tag=kind, params=npar, rel_l2={s: float(err[:, s].mean()) for s in (1, 10, 25, 50, 100)},
               rmse={s: float(rmse[s]) for s in (1, 10, 25, 50, 100)}, data_std=float(te.std()))
    print(json.dumps(res), flush=True)
    json.dump(res, open(f'{ROOT}/runs/{eq}_{kind}.json', 'w'))


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], iters=int(sys.argv[3]) if len(sys.argv) > 3 else 3000)
