"""KAFlow: u_t = ell(D) u + d(D) Phi(u, M_1 u, ..., M_J u)

ell, d, M_j : free-form complex 1D spline symbols over the wavenumber xi (Fourier multipliers)
Phi         : pointwise MultKAN (identity base + B-spline, multiplication nodes)
Time stepping: Lawson-IFRK4 (exact exponential of the linear part)
"""
import math
import torch
import torch.nn as nn


def make_knots(lo, hi, G, k):
    h = (hi - lo) / G
    return torch.arange(-k, G + k + 1, dtype=torch.float32) * h + lo


def bspline_basis(x, knots, k):
    x = x.unsqueeze(-1)
    t = knots
    b = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)
    for p in range(1, k + 1):
        b = (x - t[:-(p + 1)]) / (t[p:-1] - t[:-(p + 1)]) * b[..., :-1] \
            + (t[p + 1:] - x) / (t[p + 1:] - t[1:-p]) * b[..., 1:]
    return b


class KANLayer(nn.Module):
    """y_o = sum_i [ base[o,i] x_i + sum_g coef[o,i,g] B_g(x_i) ]"""

    def __init__(self, din, dout, G=8, k=3, lo=-3.0, hi=3.0, init_std=0.05):
        super().__init__()
        self.register_buffer('knots', make_knots(lo, hi, G, k))
        self.G, self.k, self.lo, self.hi = G, k, lo, hi
        self.coef = nn.Parameter(torch.randn(dout, din, G + k) * init_std * 0.2)
        self.base = nn.Parameter(torch.randn(dout, din) * init_std * 4)

    def forward(self, x):
        xc = x.clamp(self.lo, self.hi - 1e-5)
        B = bspline_basis(xc, self.knots, self.k)
        y = torch.einsum('...ig,oig->...o', B, self.coef)
        return y + x @ self.base.T

    def reg(self):
        return self.base.abs().sum() + self.coef.abs().mean(-1).sum()


class PhiNet(nn.Module):
    """Pointwise MultKAN: [din] -> (H multiplication nodes + A addition nodes) -> scalar"""

    def __init__(self, din, H=4, A=2, G=8, lo=-4.0, hi=4.0):
        super().__init__()
        self.H, self.A = H, A
        self.l1 = KANLayer(din, 2 * H + A, G=G, lo=lo, hi=hi)
        self.l2 = KANLayer(H + A, 1, G=G, lo=-8.0, hi=8.0)

    def forward(self, x):
        h = self.l1(x)
        prod = h[..., :self.H] * h[..., self.H:2 * self.H]
        return self.l2(torch.cat([prod, h[..., 2 * self.H:]], -1)).squeeze(-1)

    def reg(self):
        return self.l1.reg() + self.l2.reg()


class SepPhi(nn.Module):
    """Separable product Phi: sum_i psi_i(x_i) + sum_h w_h * prod_i phi_hi(x_i).
    Each 1D function = linear base + B-spline. Each factor of a product sees a single input, so polynomial PDE terms (u*u_x, x*f_x, f^3),
    being products of univariate factors, are easy to identify, and each factor can be fitted symbolically right away."""

    def __init__(self, din, H=2, G=8, lo=-4.0, hi=4.0):
        super().__init__()
        self.din, self.H, self.G, self.k, self.lo, self.hi = din, H, G, 3, lo, hi
        self.register_buffer('knots', make_knots(lo, hi, G, 3))
        nb = G + 3
        self.a = nn.Parameter(1.0 + 0.1 * torch.randn(H, din))           # constant term of each factor (initialized near 1 so that the products do not vanish)
        self.b = nn.Parameter(0.1 * torch.randn(H, din))                  # linear base
        self.c = nn.Parameter(0.02 * torch.randn(H, din, nb))             # spline
        self.w = nn.Parameter(0.1 * torch.randn(H))
        self.pb = nn.Parameter(0.05 * torch.randn(din))                   # additive branch: linear + spline
        self.pc = nn.Parameter(0.02 * torch.randn(din, nb))

    def forward(self, x):                                                 # (..., din) -> (...)
        xc = x.clamp(self.lo, self.hi - 1e-5)
        Bs = bspline_basis(xc, self.knots, self.k)                        # (..., din, nb)
        fac = self.a + self.b * x.unsqueeze(-2) + torch.einsum('...ig,hig->...hi', Bs, self.c)   # (..., H, din)
        prod = fac.prod(-1)                                               # (..., H)
        add = (self.pb * x).sum(-1) + torch.einsum('...ig,ig->...', Bs, self.pc)
        return prod @ self.w + add

    def reg(self):
        return self.c.abs().mean() + self.pc.abs().mean()


class Symbol(nn.Module):
    """nout complex symbols m(xi) = exp(log_amp) * (spline(s) + slope*s),  s = xi/xi_max in [0,1]"""

    def __init__(self, nout, xi_max, G=24, k=3, log_amp=0.0, init_std=0.3):
        super().__init__()
        self.xi_max = float(xi_max)
        self.register_buffer('knots', make_knots(0.0, 1.0, G, k))
        self.k = k
        self.coef = nn.Parameter(torch.randn(nout, 2, G + k) * init_std)
        self.slope = nn.Parameter(torch.randn(nout, 2) * init_std)
        self.log_amp = nn.Parameter(torch.full((nout, 1, 1), float(log_amp)))

    def forward(self, xi):
        s = (xi / self.xi_max).clamp(0, 1)
        B = bspline_basis(s.clamp(max=1 - 1e-6), self.knots, self.k)   # (K, G+k)
        y = torch.einsum('kg,org->okr', B, self.coef) + s[None, :, None] * self.slope[:, None, :]
        y = y * self.log_amp.exp()
        return torch.complex(y[..., 0], y[..., 1])                     # (nout, K)


class AtomSymbol(nn.Module):
    """Power-law symbol: m(xi) = sum_q (a_q + i b_q) * exp(log_s_q) * (xi/xi_max)^{alpha_q}.
    alpha_q is learned (real exponents allow fractional derivatives). Low-wavenumber data determine the high-wavenumber values (unlike a spline, it extrapolates).
    log_s tracks the large dynamic range of the coefficients (ξ^4 and so on) quickly."""

    def __init__(self, nout, xi_max, Q=4, alpha_init=(1.0, 2.0, 3.0, 4.0), jitter=0.05, c_init=0.1, s_init=0.0):
        super().__init__()
        self.xi_max = float(xi_max)
        a0 = torch.tensor(list(alpha_init)[:Q], dtype=torch.float32)
        self.alpha = nn.Parameter(a0[None].repeat(nout, 1) + jitter * torch.randn(nout, len(a0)))
        self.c = nn.Parameter(c_init * torch.randn(nout, len(a0), 2))
        self.log_s = nn.Parameter(torch.full((nout, len(a0), 1), float(s_init)))
        self.register_buffer('cmask', torch.ones(nout, len(a0), 1))

    def forward(self, xi):
        pos = xi > 0
        lx = torch.log(torch.where(pos, xi, torch.ones_like(xi)) / self.xi_max)
        atom = torch.exp(self.alpha[:, :, None] * lx[None, None, :]) * pos.float()      # (nout,Q,K)
        cc = self.c * self.cmask * self.log_s.exp()
        y = torch.einsum('oqk,oqr->okr', atom, cc)
        return torch.complex(y[..., 0], y[..., 1])


class KAFlow(nn.Module):
    def __init__(self, N, L, dt_obs, J=2, H=4, A=2, G_sym=24, G_phi=8, log_amp_ell=None,
                 log_amp_d=None, log_amp_m=None, sym='spline', Q=4):
        super().__init__()
        self.N, self.L, self.dt_obs, self.J = N, L, dt_obs, J
        k = torch.arange(N // 2 + 1, dtype=torch.float32)
        xi = 2 * math.pi * k / L
        self.register_buffer('xi', xi)
        self.register_buffer('mask', (k <= N // 3).float())
        xm = float(xi[-1])
        la = lambda v: math.log(1.0 / dt_obs) if v is None else v
        self.proj_lin = sym in ('hybrid', 'splineproj')
        if sym == 'hybrid':
            self.ell = AtomSymbol(1, xm, 5, alpha_init=(0.0, 1.0, 2.0, 3.0, 4.0), jitter=0.0, c_init=0.0)
            self.d = Symbol(1, xm, G_sym, log_amp=la(log_amp_d))
            self.M = Symbol(J, xm, G_sym, log_amp=(0.0 if log_amp_m is None else log_amp_m)) if J > 0 else None
        elif sym == 'atoms':
            self.ell = AtomSymbol(1, xm, Q, c_init=0.0)
            self.d = AtomSymbol(1, xm, Q, c_init=0.1, s_init=1.0)
            self.M = AtomSymbol(J, xm, Q, c_init=0.1, s_init=1.0) if J > 0 else None
        else:
            self.ell = Symbol(1, xm, G_sym, log_amp=la(log_amp_ell))
            self.d = Symbol(1, xm, G_sym, log_amp=la(log_amp_d))
            self.M = Symbol(J, xm, G_sym, log_amp=(0.0 if log_amp_m is None else log_amp_m)) if J > 0 else None
        self.phi = PhiNet(1 + J, H=H, A=A, G=G_phi)

    def symbols(self):
        ell = self.ell(self.xi)[0]
        # stabilization: upper bound on Re(ell*dt_sub)
        ell = torch.complex(ell.real.clamp(max=30.0 / self.dt_obs), ell.imag)
        d = self.d(self.xi)[0]
        M = self.M(self.xi) if self.J > 0 else None
        return ell, d, M

    def nonlin(self, uh, d, M):
        u = torch.fft.irfft(uh, n=self.N, dim=-1)
        feats = [u.unsqueeze(-1)]
        if self.J > 0:
            z = torch.fft.irfft(M[None] * uh[:, None, :], n=self.N, dim=-1)   # (B,J,N)
            feats.append(z.transpose(1, 2))
        f = torch.cat(feats, -1)                                              # (B,N,1+J)
        ph = self.phi(f)
        if self.proj_lin:        # the linear part of Phi (slope at zero) is moved into ell → gauge fixing, no double counting
            outer = torch.is_grad_enabled()
            with torch.enable_grad():
                z0 = torch.zeros(1, f.shape[-1], dtype=f.dtype, device=f.device, requires_grad=True)
                y0 = self.phi(z0)
                g = torch.autograd.grad(y0.sum(), z0, create_graph=outer)[0][0]
            if not outer:
                g, y0 = g.detach(), y0.detach()
            ph = ph - (f * g).sum(-1) - y0.squeeze()
        return d * torch.fft.rfft(ph, dim=-1) * self.mask

    def step(self, uh, dt, syms):
        ell, d, M = syms
        E = torch.exp(ell * dt / 2)
        E2 = E * E
        a = self.nonlin(uh, d, M)
        b = self.nonlin(E * (uh + 0.5 * dt * a), d, M)
        c = self.nonlin(E * uh + 0.5 * dt * b, d, M)
        dd = self.nonlin(E2 * uh + dt * E * c, d, M)
        return E2 * uh + dt / 6 * (E2 * a + 2 * E * (b + c) + dd)

    def forward(self, u0, n_steps, sub=4, dt_obs=None):
        dt_obs = dt_obs or self.dt_obs
        syms = self.symbols()
        dt = dt_obs / sub
        uh = torch.fft.rfft(u0, dim=-1) * self.mask
        outs = [torch.fft.irfft(uh, n=self.N, dim=-1)]
        for _ in range(n_steps):
            for _ in range(sub):
                uh = self.step(uh, dt, syms)
            outs.append(torch.fft.irfft(uh, n=self.N, dim=-1))
        return torch.stack(outs, 1)

    def reg(self):
        return self.phi.reg()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
