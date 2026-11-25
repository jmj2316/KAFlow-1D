"""KAFlow-U: a single model with the same structure and the same rules for every benchmark.

Two root causes are prevented by construction.
 (1) Gauge redundancy: ell, d and the linear part of Phi represent the same linear operation → poor local minima, seed sensitivity
     → the x-independent part of the linear part of Phi is removed by projection (position-dependent linear terms such as x*f_x stay in Phi)
 (2) Spectral identifiability: a free-form spline symbol is undetermined at high wavenumbers without data energy → long-term drift
     → symbol = fixed-exponent symbolic dictionary sum_p c_p (xi/xi_ref)^p  +  spline residual with an energy-weighted penalty
        (linear in the coefficients → well conditioned; the dictionary decides the extrapolation in bands without signal)

Time stepping:   u_t = ell(D) u + d(D) Phi_proj(u, M_1 u, ..., M_J u)        (Lawson-IFRK4)
Static operator: G f = ell(D) f + Phi_proj(f, M_1 f, ..., M_J f, x)          (d = 1)
"""
import math
import torch
import torch.nn as nn
from kaf import make_knots, bspline_basis, SepPhi

INT_EXPS = (0.0, 1.0, 2.0, 3.0, 4.0)
HALF_EXPS = (0.5, 1.5, 2.5, 3.5)


def rate_stat(traj, dt):
    """Relative rate of change of the data, rms(u_{t+1}-u_t) / (dt * rms(u))  (traj: (S,T,N))"""
    du = traj[:, 1:] - traj[:, :-1]
    return float(du.pow(2).mean().sqrt() / (dt * traj.pow(2).mean().sqrt()))


def spectrum_stats(u, L):
    """Energy per wavenumber E_k and energy-weighted RMS wavenumber xi_ref of the training data (u: (..., N) real field)."""
    N = u.shape[-1]
    E = (torch.fft.rfft(u.reshape(-1, N).double(), dim=-1).abs() ** 2).mean(0)
    xi = 2 * math.pi * torch.arange(N // 2 + 1, dtype=torch.float64) / L
    Ek = E.clone(); Ek[0] = 0
    xi_ref = float(torch.sqrt((Ek * xi ** 2).sum() / Ek.sum()))
    return E.float(), max(xi_ref, float(xi[1]))


class DictSymbol(nn.Module):
    """m(xi) = scale * [ sum_p c_p (xi/xi_ref)^p  +  r(xi) ]   (complex, rfft grid xi>=0)
    c_p: fixed-exponent dictionary coefficients (L1, half-integer exponents weighted 3 times = integer exponents preferred)
    r  : spline residual with penalty weight w_k = 1/(E_k/E_max + tau) → pushed to 0 in bands without data"""

    def __init__(self, nout, xi, E, xi_ref, scale=1.0, exps=INT_EXPS + HALF_EXPS, G=16, tau=1e-3, init_std=0.0, init_max_p=1.0, maxnorm=False):
        super().__init__()
        self.scale, self.exps = float(scale), tuple(exps)
        dt_ = torch.get_default_dtype()
        # maxnorm: normalize each atom by its maximum on the grid (for multipliers whose size must be tied, such as d and M; Phi absorbs the amplitude)
        self.ref = float(xi.max()) if maxnorm else float(xi_ref)
        xs = (xi / self.ref).to(dt_)
        A = torch.stack([torch.where(xs > 0, xs.clamp_min(1e-12) ** p, torch.full_like(xs, 1.0 if p == 0 else 0.0)) for p in exps])
        self.register_buffer('A', A)                                                   # (P,K)
        s = (xi / xi.max()).clamp(0, 1 - 1e-6).to(dt_)
        self.register_buffer('B', bspline_basis(s, make_knots(0.0, 1.0, G, 3).to(dt_), 3))     # (K,G+3)
        w = 1.0 / (E / E.max() + tau)
        self.register_buffer('w', (w / w.mean()).to(dt_))
        self.register_buffer('l1w', torch.tensor([1.0 if float(p).is_integer() else 3.0 for p in exps], dtype=dt_))
        lowp = torch.tensor([1.0 if p <= init_max_p else 0.0 for p in exps], dtype=dt_)          # random initial values only on low orders (identity, derivatives)
        self.c = nn.Parameter(init_std * torch.randn(nout, len(exps), 2) * lowp[None, :, None])
        self.r = nn.Parameter(torch.zeros(nout, 2, G + 3))
        self.register_buffer('mask', torch.ones(len(exps), dtype=dt_))
        self.register_buffer('Ew', (E / E.max()).to(dt_))

    def raw(self):
        y = torch.einsum('pk,opr->okr', self.A, self.c * self.mask[None, :, None]) + torch.einsum('kg,org->okr', self.B, self.r)
        return torch.complex(y[..., 0], y[..., 1])                                     # (nout,K) normalized units

    def forward(self):
        return self.scale * self.raw()

    def pen_l1(self):
        return (self.l1w[None, :] * self.mask[None, :] * self.c.norm(dim=-1)).sum()

    def pen_res(self):
        r = torch.einsum('kg,org->okr', self.B, self.r)
        return (self.w[None, :, None] * r ** 2).mean() * 2

    @torch.no_grad()
    def sparsify(self, floor_rel=1e-3, size=None, kappa_max=1e3):
        """Sparsify the learned symbol (dictionary + residual) with the energy-weighted (sqrt(E)) BIC-optimal subset. No band threshold.
        The selected atoms are refitted by least squares for Re/Im; the residual is set to 0 and frozen. (nout=1 only)"""
        from itertools import combinations
        y = self.raw()[0].detach().cpu().to(torch.complex128)                # (K,) normalized units (kept complex)
        A = self.A.cpu().double().T                                          # (K,P)
        w = self.Ew.cpu().double().clamp_min(0).sqrt()
        P = A.shape[1]; best = (float('inf'), None)
        Y = torch.stack([y.real, y.imag], 1)                                  # (K,2)
        sig2 = (floor_rel * (w[:, None] * Y).abs().max()) ** 2            # noise floor measured in the energy-weighted space (not pulled by huge high-wavenumber values)
        n = float((w ** 2).sum() ** 2 / (w ** 4).sum())                 # effective sample size (Kish): number of band points that drive the residual
        for r in (range(0, P + 1) if size is None else [size]):     # with size given: weighted-residual-minimizing subset of that size (best-subset path)
            for S in combinations(range(P), r):
                if r == 0:
                    res = (w[:, None] * Y).pow(2).sum(); C = torch.zeros(P, 2, dtype=torch.float64)
                else:
                    As = A[:, S] * w[:, None]
                    if r > 1:                                              # exclude collinear (canceling) subsets: condition number of the column-normalized weighted design matrix
                        An = As / As.norm(dim=0, keepdim=True).clamp_min(1e-30)
                        sv = torch.linalg.svdvals(An)
                        if float(sv[0] / sv[-1].clamp_min(1e-30)) > kappa_max:
                            continue
                    cs = torch.linalg.lstsq(As, w[:, None] * Y).solution
                    res = (As @ cs - w[:, None] * Y).pow(2).sum()
                    C = torch.zeros(P, 2, dtype=torch.float64); C[list(S)] = cs
                nh = sum(1 for i in S if not float(self.exps[i]).is_integer())   # integer orders preferred (same prior as the 3-times L1 on half-integers): residual doubled per half-integer atom
                res = res * (2.0 ** nh)
                bic = (n * math.log(float(res) / float((w ** 2).sum()) + float(sig2) / float((w ** 2).mean())) + 2 * r * math.log(max(n, 2.0))) if size is None else float(res)
                if bic < best[0]:
                    best = (bic, C, S)
        _, C, S = best
        m = torch.zeros(P); m[list(S)] = 1.0
        self.mask.copy_(m.to(self.mask)); self.c.data[0] = C.to(self.c)
        self.r.data.zero_(); self.r.requires_grad_(False)
        return [self.exps[i] for i in S]

    def physical(self, xi_ref=None):
        """Dictionary coefficients in physical units: m(xi) ≈ sum_p C_p xi^p,  C_p = scale * c_p / ref^p"""
        c = self.c.detach().cpu(); ref = self.ref
        return {p: [round(float(self.scale * c[o, i, 0] / ref ** p), 5), round(float(self.scale * c[o, i, 1] / ref ** p), 5)]
                for i, p in enumerate(self.exps) for o in range(c.shape[0])}


class ProjPhi(nn.Module):
    """Phi_proj(feats, x) = Phi(feats, x) - sum_j gbar_j feats_j,  gbar_j = mean_x dPhi/dfeat_j |_{feats=0}
    Only the x-independent linear part is removed → no gauge redundancy with ell; position-dependent linear terms (x f_x) are kept."""

    def __init__(self, nfeat, use_x, H=2, G=8):
        super().__init__()
        self.nfeat, self.use_x = nfeat, use_x
        self.phi = SepPhi(nfeat + (1 if use_x else 0), H=H, G=G)

    def forward(self, feats, x=None):                                                 # feats (B,N,F), x (N,)
        inp = torch.cat([feats, x[None, :, None].expand(feats.shape[0], -1, 1)], -1) if self.use_x else feats
        out = self.phi(inp)
        outer = torch.is_grad_enabled()
        with torch.enable_grad():
            if self.use_x:
                z0 = torch.zeros(x.shape[0], self.nfeat, device=feats.device, dtype=feats.dtype, requires_grad=True)
                y0 = self.phi(torch.cat([z0, x[:, None]], -1))
            else:
                z0 = torch.zeros(1, self.nfeat, device=feats.device, dtype=feats.dtype, requires_grad=True)
                y0 = self.phi(z0)
            g = torch.autograd.grad(y0.sum(), z0, create_graph=outer)[0].mean(0)        # slope averaged over x (F,)
        if not outer:
            g = g.detach()
        return out - (feats * g).sum(-1)

    def reg(self):
        return self.phi.reg()


class KAFlowU(nn.Module):
    """Time-stepping generator. The symbol scale is set from data statistics (E, xi_ref) (scale = 1/dt)."""

    def __init__(self, N, L, dt_obs, E, xi_ref, rate=None, J=2, H=2, G_phi=8):
        super().__init__()
        self.N, self.L, self.dt_obs, self.J, self.xi_ref = N, L, dt_obs, J, xi_ref
        k = torch.arange(N // 2 + 1, dtype=torch.float32)
        xi = 2 * math.pi * k / L
        self.register_buffer('xi', xi)
        self.register_buffer('mask', (k <= N // 3).float())
        sc = rate if rate is not None else 1.0 / dt_obs        # scale = relative rate of change of the data (common rule for all problems)
        LOW = (0.0, 1.0, 2.0)                                  # d and M use p<=2 (high orders only in ell, which is integrated exactly)
        self.ell = DictSymbol(1, xi, E, xi_ref, scale=sc)
        self.d = DictSymbol(1, xi, E, xi_ref, scale=sc, exps=(0.0, 1.0), init_std=0.3)        # flux = identity or first derivative
        self.M = DictSymbol(J, xi, E, xi_ref, scale=1.0, exps=LOW, init_std=0.5)
        self.phi = ProjPhi(1 + J, use_x=False, H=H, G=G_phi)

    def symbols(self):
        ell = self.ell()[0]
        return torch.complex(ell.real.clamp(max=3.0 / self.dt_obs), ell.imag), self.d()[0], self.M()      # growth per step ≤ e^3

    def nonlin(self, uh, d, M):
        u = torch.fft.irfft(uh, n=self.N, dim=-1)
        z = torch.fft.irfft(M[None] * uh[:, None, :], n=self.N, dim=-1).transpose(1, 2)
        feats = torch.cat([u.unsqueeze(-1), z], -1)
        return d * torch.fft.rfft(self.phi(feats), dim=-1) * self.mask

    def step(self, uh, dt, syms):
        ell, d, M = syms
        E = torch.exp(ell * dt / 2); E2 = E * E
        a = self.nonlin(uh, d, M)
        b = self.nonlin(E * (uh + 0.5 * dt * a), d, M)
        c = self.nonlin(E * uh + 0.5 * dt * b, d, M)
        dd = self.nonlin(E2 * uh + dt * E * c, d, M)
        return E2 * uh + dt / 6 * (E2 * a + 2 * E * (b + c) + dd)

    def forward(self, u0, n_steps, sub=4):
        syms = self.symbols(); dt = self.dt_obs / sub
        uh = torch.fft.rfft(u0, dim=-1) * self.mask
        outs = [torch.fft.irfft(uh, n=self.N, dim=-1)]
        for _ in range(n_steps):
            for _ in range(sub):
                uh = self.step(uh, dt, syms)
            outs.append(torch.fft.irfft(uh, n=self.N, dim=-1))
        return torch.stack(outs, 1)

    def penalty(self, lam_l1, lam_res):
        s = [self.ell, self.d, self.M]
        return lam_l1 * sum(m.pen_l1() for m in s) + lam_res * sum(m.pen_res() for m in s)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class KAFlowUOp(nn.Module):
    """Static operator G f = ell(D) f + Phi_proj(f, M f, x)   (d = 1, uses the position input)"""

    def __init__(self, N, L, E, xi_ref, scale, J=1, H=2, G_phi=8):
        super().__init__()
        self.N, self.L, self.J, self.xi_ref = N, L, J, xi_ref
        k = torch.arange(N // 2 + 1, dtype=torch.float32)
        xi = 2 * math.pi * k / L
        self.register_buffer('xi', xi)
        self.register_buffer('x', -L / 2 + torch.arange(N, dtype=torch.float32) * L / N)
        self.ell = DictSymbol(1, xi, E, xi_ref, scale=scale)
        self.M = DictSymbol(J, xi, E, xi_ref, scale=1.0, exps=(0.0, 1.0, 2.0), init_std=0.5) if J > 0 else None
        self.phi = ProjPhi(1 + J, use_x=True, H=H, G=G_phi)

    def forward(self, f):
        fh = torch.fft.rfft(f, dim=-1)
        out = torch.fft.irfft(self.ell()[0] * fh, n=self.N, dim=-1)
        feats = [f.unsqueeze(-1)]
        if self.J > 0:
            feats.append(torch.fft.irfft(self.M()[None] * fh[:, None, :], n=self.N, dim=-1).transpose(1, 2))
        return out + self.phi(torch.cat(feats, -1), self.x)

    def penalty(self, lam_l1, lam_res):
        s = [self.ell] + ([self.M] if self.J > 0 else [])
        return lam_l1 * sum(m.pen_l1() for m in s) + lam_res * sum(m.pen_res() for m in s)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class XFn(nn.Module):
    """1D function of position a(x) = linear + B-spline, centered to zero mean (ell carries the constant component: gauge fixing)."""

    def __init__(self, nout, x, G=12, init_std=0.01):
        super().__init__()
        lo, hi = float(x.min()) - 0.1, float(x.max()) + 0.1
        self.register_buffer('B', bspline_basis(x.clamp(lo, hi - 1e-6), make_knots(lo, hi, G, 3), 3))   # (N,G+3)
        self.register_buffer('x', x)
        self.c = nn.Parameter(init_std * torch.randn(nout, G + 3))
        self.b = nn.Parameter(init_std * torch.randn(nout))

    def forward(self):
        a = self.c @ self.B.T + self.b[:, None] * self.x[None]                          # (nout,N)
        return a - a.mean(-1, keepdim=True)


class FullProjPhi(nn.Module):
    """Removes the whole linear part of Phi including its x dependence: Phi(feats,x) - Phi(0,x) - sum_j g_j(x) feats_j  → only the pure nonlinearity remains.
    (linear position-dependent terms are carried by the low-rank symbol alone → no gauge redundancy)"""

    def __init__(self, nfeat, H=2, G=8):
        super().__init__()
        self.nfeat = nfeat
        self.phi = SepPhi(nfeat + 1, H=H, G=G)

    def forward(self, feats, x):
        Bn = feats.shape[0]
        out = self.phi(torch.cat([feats, x[None, :, None].expand(Bn, -1, 1)], -1))
        outer = torch.is_grad_enabled()
        with torch.enable_grad():
            z0 = torch.zeros(x.shape[0], self.nfeat, device=feats.device, dtype=feats.dtype, requires_grad=True)
            y0 = self.phi(torch.cat([z0, x[:, None]], -1))                               # (N,)
            g = torch.autograd.grad(y0.sum(), z0, create_graph=outer)[0]                 # (N,F) slope at each position
        if not outer:
            g, y0 = g.detach(), y0.detach()
        return out - y0[None] - (feats * g[None]).sum(-1)

    def reg(self):
        return self.phi.reg()


class KAFlowUOp2(nn.Module):
    """Static operator + position-dependent low-rank linear symbol:
       G f = ell(D) f + sum_r a_r(x) (m_r(D) f) + Phi_nl(f, M f, x)
    represents position-dependent linear terms such as x*f_x (a=x, m=i xi) and x^2 f (a=x^2, m=1) exactly with a linear structure. Cost O(R N log N)."""

    def __init__(self, N, L, E, xi_ref, scale, R=2, J=1, H=2, G_phi=8):
        super().__init__()
        self.N, self.L, self.J, self.R, self.xi_ref = N, L, J, R, xi_ref
        k = torch.arange(N // 2 + 1, dtype=torch.float32)
        xi = 2 * math.pi * k / L
        self.register_buffer('xi', xi)
        self.register_buffer('x', -L / 2 + torch.arange(N, dtype=torch.float32) * L / N)
        self.ell = DictSymbol(1, xi, E, xi_ref, scale=scale)
        self.mr = DictSymbol(R, xi, E, xi_ref, scale=1.0, exps=(0.0, 1.0, 2.0), init_std=0.3)
        self.ar = XFn(R, self.x)
        self.ar_scale = scale
        self.M = DictSymbol(J, xi, E, xi_ref, scale=1.0, exps=(0.0, 1.0, 2.0), init_std=0.5) if J > 0 else None
        self.phi = FullProjPhi(1 + J, H=H, G=G_phi)

    def forward(self, f):
        fh = torch.fft.rfft(f, dim=-1)
        out = torch.fft.irfft(self.ell()[0] * fh, n=self.N, dim=-1)
        mf = torch.fft.irfft(self.mr()[None] * fh[:, None, :], n=self.N, dim=-1)       # (B,R,N)
        out = out + self.ar_scale * (self.ar()[None] * mf).sum(1)
        feats = [f.unsqueeze(-1)]
        if self.J > 0:
            feats.append(torch.fft.irfft(self.M()[None] * fh[:, None, :], n=self.N, dim=-1).transpose(1, 2))
        return out + self.phi(torch.cat(feats, -1), self.x)

    def penalty(self, lam_l1, lam_res):
        s = [self.ell, self.mr] + ([self.M] if self.J > 0 else [])
        return lam_l1 * sum(m.pen_l1() for m in s) + lam_res * sum(m.pen_res() for m in s)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
