"""KAFlow-Q: Hermitian special form for quantum dynamics.  i dpsi/dt = [ T(D) + v(x) + n(|psi|^2) ] psi
  T(xi)  : even function of the wavenumber (free-form spline or power-law atoms)
  v(x)   : free-form 1D function of position
  n(rho) : free-form 1D function of the density rho=|psi|^2 (nonlinear term; inactive for linear problems)
Strang split step exp(-i dt/2 V) FFT exp(-i dt T) iFFT exp(-i dt/2 V), exactly unitary, with norm renormalization.
Continuous in time with arbitrary dt; the position-dependent and nonlocal (wavenumber) components are separate, so symbolic fitting is immediate."""
import math
import torch
import torch.nn as nn
import sys
import os
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from kaf import make_knots, bspline_basis
from kafu import DictSymbol, INT_EXPS, HALF_EXPS


class Fn1D(nn.Module):
    """1D function = linear base + constant + B-spline (linear extrapolation outside the range). Output scale amp fixed."""

    def __init__(self, lo, hi, G=12, amp=1.0, init_std=0.02):
        super().__init__()
        self.lo, self.hi, self.G, self.amp = lo, hi, G, amp
        self.register_buffer('knots', make_knots(lo, hi, G, 3).to(torch.get_default_dtype()))
        self.c = nn.Parameter(init_std * torch.randn(G + 3))
        self.b = nn.Parameter(torch.zeros(()))
        self.a = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        xc = x.clamp(self.lo, self.hi - 1e-6)
        B = bspline_basis(xc, self.knots, 3).to(self.c.dtype)
        return self.amp * (self.a + self.b * x + B @ self.c)


class PosDictT(nn.Module):
    """Kinetic energy T(|xi|) = scale * sum_p c_p (|xi|/xi_ref)^p,  c_p = softplus(theta_p) >= 0,  p in {0.5..4} (no constant term).
    T(0)=0, T>=0, monotonically increasing in |xi| (dispersion relation of a massive particle). → removes the H/-H mirror solution; the constant (global phase) goes to v."""

    def __init__(self, k_abs, E, xi_ref, scale, exps=(1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5)):
        super().__init__()
        dt_ = torch.get_default_dtype()
        self.scale, self.exps, self.ref = float(scale), tuple(exps), float(xi_ref)
        xs = (k_abs / xi_ref).to(dt_)
        self.register_buffer('A', torch.stack([xs.clamp_min(0) ** p for p in exps]))
        self.register_buffer('Ew', (E / E.max()).to(dt_))
        self.register_buffer('l1w', torch.tensor([1.0 if float(p).is_integer() else 3.0 for p in exps], dtype=dt_))
        self.register_buffer('mask', torch.ones(len(exps), dtype=dt_))
        self.theta = nn.Parameter(torch.full((len(exps),), -3.0, dtype=dt_))

    def coef(self):
        return nn.functional.softplus(self.theta) * self.mask

    def forward(self):
        return self.scale * (self.coef() @ self.A)

    def pen_l1(self):
        return (self.l1w * self.coef()).sum()

    @torch.no_grad()
    def sparsify(self, floor_rel=1e-3, size=None, kappa_max=1e3):
        """Nonnegative least-squares (NNLS) subset selection: same candidate rules as DictSymbol.sparsify (BIC floor / by size, condition number, integer preference)."""
        from itertools import combinations
        from scipy.optimize import nnls
        y = (self.coef() @ self.A).cpu().double().numpy()
        A = self.A.cpu().double().T.numpy(); w = self.Ew.cpu().double().clamp_min(0).sqrt().numpy()
        P = A.shape[1]; best = (float('inf'), None, ())
        sig2 = (floor_rel * abs(w * y).max()) ** 2; W2 = float((w ** 2).sum()); n = float((w ** 2).sum() ** 2 / (w ** 4).sum())
        import numpy as np
        for r in (range(1, P + 1) if size is None else [size]):
            for S in combinations(range(P), r):
                As = A[:, S] * w[:, None]
                if r > 1:
                    An = As / np.maximum(np.linalg.norm(As, axis=0, keepdims=True), 1e-30)
                    sv = np.linalg.svd(An, compute_uv=False)
                    if sv[0] / max(sv[-1], 1e-30) > kappa_max:
                        continue
                cs, _ = nnls(As, w * y)
                res = float(((As @ cs - w * y) ** 2).sum()) * (2.0 ** sum(1 for i in S if not float(self.exps[i]).is_integer()))
                crit = (n * math.log(res / W2 + sig2 / float((w ** 2).mean())) + r * math.log(max(n, 2.0))) if size is None else res
                if crit < best[0]:
                    C = np.zeros(P); C[list(S)] = cs; best = (crit, C, S)
        _, C, S = best
        S = [i for i in S if C[i] > 0]
        m = torch.zeros(P, dtype=self.mask.dtype); m[S] = 1.0
        self.mask.copy_(m.to(self.mask))
        Ct = torch.tensor(C, dtype=self.theta.dtype).clamp_min(1e-12)
        self.theta.data.copy_(torch.log(torch.expm1(Ct)).to(self.theta))
        return [self.exps[i] for i in S]

    def physical(self, xi_ref=None):
        c = self.coef().detach().cpu()
        return {p: [round(float(self.scale * c[i] / self.ref ** p), 5), 0.0] for i, p in enumerate(self.exps)}


class KAFlowQ(nn.Module):
    def __init__(self, N=128, L=4.0, dt=1e-4, nonlinear=False, G=12, tmode='spline', E=None, xi_ref=None, tscale=None):
        super().__init__()
        self.N, self.L, self.dt, self.nonlinear, self.tmode = N, L, dt, nonlinear, tmode
        dx = L / N
        self.register_buffer('x', -L / 2 + torch.arange(N, dtype=torch.get_default_dtype()) * dx)
        k = 2 * math.pi * torch.fft.fftfreq(N, d=dx)
        self.register_buffer('k', k.to(torch.get_default_dtype()))
        kmax = float(k.abs().max())
        self.kmax = kmax
        # T(|xi|): spline in s=|xi|/kmax (scaled by amp = kmax^2/2 → O(1) coefficients)  or c*|xi|^alpha
        if tmode == 'dict':                                   # KAFlow-U rule: fixed-exponent dictionary + energy-weighted residual (real part only = Hermitian)
            self.Td = DictSymbol(1, k.abs().to(torch.get_default_dtype()), E, xi_ref, scale=tscale)
        elif tmode == 'dictpos':                              # nonnegative, monotone kinetic-energy dictionary (removes the H/-H mirror solution)
            self.Td = PosDictT(k.abs().to(torch.get_default_dtype()), E, xi_ref, scale=tscale)
        elif tmode == 'power':
            self.alpha = nn.Parameter(torch.tensor(2.0) + 0.05 * torch.randn(()))
            self.logc = nn.Parameter(torch.tensor(math.log(0.5 * kmax ** 2)) + 0.0)       # c*(|xi|/kmax)^alpha
        else:
            self.T = Fn1D(0.0, 1.0, G, amp=0.5 * kmax ** 2)
        self.v = Fn1D(-L / 2 - 0.1, L / 2 + 0.1, G, amp=1.0)
        if nonlinear:
            if tmode in ('dict', 'dictpos'):                  # n(rho) = c1 rho + c2 rho^2, no constant term (removes the gauge with the constant of v = global phase)
                self.nc = nn.Parameter(torch.zeros(2))
            else:
                self.n = Fn1D(0.0, 0.5, G, amp=1.0)

    def Tk(self):
        if self.tmode == 'dict':
            return self.Td()[0].real
        if self.tmode == 'dictpos':
            return self.Td()
        s = self.k.abs() / self.kmax
        if self.tmode == 'power':
            return self.logc.exp() * torch.where(s > 0, s.clamp_min(1e-12), torch.ones_like(s)).pow(self.alpha) * (s > 0).float()
        return self.T(s)

    def potential(self, psi):
        V = self.v(self.x)
        if self.nonlinear:
            rho = psi.abs() ** 2
            V = V + (self.nc[0] * rho + self.nc[1] * rho ** 2 if self.tmode in ('dict', 'dictpos') else self.n(rho))
        return V

    def step(self, psi):
        dt = self.dt
        psi = psi * torch.exp(-0.5j * dt * self.potential(psi))
        psi = torch.fft.ifft(torch.fft.fft(psi) * torch.exp(-1j * dt * self.Tk()))
        psi = psi * torch.exp(-0.5j * dt * self.potential(psi))
        return psi / psi.abs().pow(2).sum(-1, keepdim=True).sqrt()

    def forward(self, psi0, n_steps):                       # (B,N) complex -> (B,n_steps+1,N)
        out = [psi0]; psi = psi0
        for _ in range(n_steps):
            psi = self.step(psi); out.append(psi)
        return torch.stack(out, 1)

    def penalty(self, lam1, lamr):
        if self.tmode == 'dictpos':
            return lam1 * self.Td.pen_l1()
        return lam1 * self.Td.pen_l1() + lamr * self.Td.pen_res() if self.tmode == 'dict' else 0.0

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
