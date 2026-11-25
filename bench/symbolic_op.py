"""Automatic symbolic distillation of position-dependent linear operators (the steps of KANO_symbolic without human intervention).
 1) Sparse approximation of the position-dependent part K(x,xi) = sum_r scale a_r(x) m_r(xi) of the trained KAFlowUOp2
    on the monomial dictionary (x^n - <x^n>)(i xi)^p, n=1..4, p=0..2 → candidate supports (BIC floor path + by size, collinear sets excluded)
 2) For each candidate, fix the functional form and retrain the coefficients (and ell, Phi) (Adam + L-BFGS)
 3) Group A held-out validation (sample) loss, smallest support within 1.1 times the best
 Result: coefficients in original units, reported as G f = sum c_{n,p} x^n d^p f (+ Phi_nl).
 python -B symbolic_op.py 1 0
"""
import sys, os, json, math, copy, itertools
import numpy as np
import torch
import torch.nn as nn
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from kafu import KAFlowUOp2, spectrum_stats

torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))
R = ROOT
g, seed = int(sys.argv[1]), int(sys.argv[2])
ld = lambda p: torch.load(p, weights_only=False, map_location='cpu')


def data(nm):
    d = torch.load(f'{R}/ext/KANO/G_{g}/dataset/{nm}_{g}.pt', weights_only=False, map_location='cpu')
    return torch.stack([e[1] for e in d]).float(), torch.stack([e[2] for e in d]).float()


Ftr, Gtr = data('train'); Fva, Gva = data('sample'); Fte, Gte = data('test')
N, L = 128, 2 * math.pi
E, xi_ref = spectrum_stats(Ftr, L)
scale = float(Gtr.pow(2).mean().sqrt() / Ftr.pow(2).mean().sqrt())
base = KAFlowUOp2(N, L, E, xi_ref, scale, R=2, J=1, H=2)
base.load_state_dict(ld(f'{R}/runs/synth/G{g}_U2op_s{seed}.pt')); base.eval()
rel = lambda p, t: (p - t).norm(dim=1) / t.norm(dim=1)

x = base.x.double(); xi = base.xi.double(); ik = 1j * xi
ATOMS = [(n, p) for n in range(1, 5) for p in range(3)]
XC = {n: x ** n - (x ** n).mean() for n in range(1, 5)}                       # zero-mean monomials (ell carries the constant: same gauge)

# --- 1) candidate sparse approximations of K(x,xi)
with torch.no_grad():
    K = (base.ar_scale * base.ar().double()[:, :, None] * base.mr().to(torch.complex128)[:, None, :]).sum(0)   # (N_x, K)
w = (E.double() / E.max()).sqrt()                                             # wavenumber energy weights (uniform in position: Group A covers the whole domain)
cols = [(XC[n][:, None] * ik[None, :] ** p) for (n, p) in ATOMS]
Acols = torch.stack([(c * w[None, :]).reshape(-1) for c in cols], 1)           # (N_x*K, P) complex
Y = (K * w[None, :]).reshape(-1)
cn = Acols.abs().pow(2).sum(0).sqrt(); An = Acols / cn                       # column normalization
P = len(ATOMS); nn_eff = float((w ** 2).sum() ** 2 / (w ** 4).sum()) * N


def fit(S):
    As = An[:, list(S)]
    c = torch.linalg.lstsq(As, Y[:, None]).solution[:, 0]
    return c, float((As @ c - Y).abs().pow(2).sum())


def cands():
    out = set()
    sig_base = float(Y.abs().max())
    for fl in (1e-3, 3e-3, 1e-2, 3e-2):
        best = (float('inf'), ())
        for r in range(1, P + 1):
            for S in itertools.combinations(range(P), r):
                if r > 1:
                    sv = torch.linalg.svdvals(An[:, list(S)])
                    if float(sv[0] / sv[-1].clamp_min(1e-30)) > 1e3:
                        continue
                _, res = fit(S)
                bic = nn_eff * math.log(res / float(Y.abs().pow(2).sum()) + (fl * sig_base) ** 2 / float(Y.abs().pow(2).mean())) + 2 * r * math.log(nn_eff)
                if bic < best[0]:
                    best = (bic, S)
            if r >= 4:                                                        # combinations of more than 4 terms are skipped (12C5 and beyond are not worth the cost)
                break
        out.add(tuple(best[1]))
    for size in (1, 2, 3):
        best = (float('inf'), ())
        for S in itertools.combinations(range(P), size):
            if size > 1:
                sv = torch.linalg.svdvals(An[:, list(S)])
                if float(sv[0] / sv[-1].clamp_min(1e-30)) > 1e3:
                    continue
            _, res = fit(S)
            if res < best[0]:
                best = (res, S)
        out.add(tuple(best[1]))
    return [s for s in out if len(s) > 0]


class Distilled(nn.Module):
    """G f = ell(D) f + sum_{(n,p) in S} c_np (x^n - <x^n>) d^p f + Phi_nl   (ell, Phi_nl, M are copied from the base model and retrained)"""

    def __init__(self, S, c0, use_phi=True):
        super().__init__()
        self.use_phi = use_phi
        self.b = copy.deepcopy(base)
        with torch.no_grad():                                                               # ell keeps only the sparse dictionary (residual fixed at 0) → fully symbolic model
            self.b.ell.r.zero_()
        self.b.ell.r.requires_grad_(False)
        for mod in (self.b.mr, self.b.ar):                                                  # the low-rank part replaced by the distillation is not used
            for p_ in mod.parameters():
                p_.requires_grad_(False)
        self.S = [ATOMS[i] for i in S]
        self.register_buffer('Xc', torch.stack([XC[n].float() for n, _ in self.S]))         # (s,N)
        self.register_buffer('Pw', torch.tensor([p for _, p in self.S]))
        c0 = c0 / cn[list(S)].to(c0)                                                       # undo the normalization → coefficients in original units
        self.c = nn.Parameter(torch.stack([c0.real, c0.imag], -1).float())

    def coef(self):
        return torch.complex(self.c[:, 0], self.c[:, 1])

    def forward(self, f):
        b = self.b
        fh = torch.fft.rfft(f, dim=-1)
        out = torch.fft.irfft(b.ell()[0] * fh, n=N, dim=-1)
        ikx = (1j * b.xi).to(torch.complex128 if f.dtype == torch.float64 else torch.complex64)
        ikf = torch.stack([torch.ones_like(ikx) if int(pp) == 0 else ikx ** int(pp) for pp in self.Pw])   # (s,K): avoids 0^0 NaN
        Dp = torch.fft.irfft(ikf[None] * fh[:, None, :], n=N, dim=-1)                       # (B,s,N)
        cc = self.coef()
        out = out + (cc.real[None, :, None] * self.Xc[None] * Dp).sum(1)                    # real field: only the real part of the coefficients matters
        if not self.use_phi:                                                                # purely linear candidate (for the parsimony comparison)
            return out
        feats = [f.unsqueeze(-1), torch.fft.irfft(b.M()[None] * fh[:, None, :], n=N, dim=-1).transpose(1, 2)]
        return out + b.phi(torch.cat(feats, -1), b.x)


def train(m):
    ps = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(ps, lr=2e-3)
    for it in range(1000):
        lr = 2e-3 * 0.5 * (1 + math.cos(math.pi * it / 1000)) + 1e-5
        for gp in opt.param_groups:
            gp['lr'] = lr
        bi = torch.randint(0, Ftr.shape[0], (64,))
        loss = rel(m(Ftr[bi]), Gtr[bi]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    lb = torch.optim.LBFGS(ps, lr=1.0, max_iter=300, history_size=50, line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-15)

    def cl():
        lb.zero_grad(); l = rel(m(Ftr), Gtr).mean(); l.backward(); return l
    lb.step(cl)
    return m


trials = {}
for S in cands():
    c0, _ = fit(S)
    for use_phi in (True, False):
        m = train(Distilled(S, c0, use_phi))
        with torch.no_grad():
            v = float(rel(m(Fva), Gva).mean())
        trials[(tuple(S), use_phi)] = (v if v == v else float('inf'), m)
        print(f'  support {[ATOMS[i] for i in S]} Phi={use_phi} → validation {v:.3e}', flush=True)
vmin = min(v for v, _ in trials.values())
key_sel = min([k_ for k_, (v, _) in trials.items() if v <= 1.1 * vmin], key=lambda k_: (len(k_[0]) + (1 if k_[1] else 0), trials[k_][0]))
Ssel, phi_sel = key_sel
m = trials[key_sel][1]
# (3) final polishing of the coefficients: float64 full-batch L-BFGS 2000 (converges weakly identified terms as well)
m = m.double(); Ftr, Gtr, Fva, Gva, Fte, Gte = (t.double() for t in (Ftr, Gtr, Fva, Gva, Fte, Gte))
for mod in m.modules():
    for nm_, buf in list(mod.named_buffers(recurse=False)):
        if buf.is_floating_point():
            setattr(mod, nm_, buf.double())
ps = [p for p in m.parameters() if p.requires_grad]
lb = torch.optim.LBFGS(ps, lr=1.0, max_iter=2000, history_size=100, line_search_fn='strong_wolfe', tolerance_grad=1e-14, tolerance_change=1e-18)
def cl():
    lb.zero_grad(); l = rel(m(Ftr), Gtr).mean(); l.backward(); return l
lb.step(cl)

# --- coefficients in original units: c_np for n>=1, and ell(physical) - sum_n c_np <x^n> for n=0
with torch.no_grad():
    te = float(rel(m(Fte), Gte).mean()); trn = float(rel(m(Ftr), Gtr).mean()); va = float(rel(m(Fva), Gva).mean())
    coef = {}
    cc = m.coef().real.double()
    for (n, p), c in zip(m.S, cc):
        coef[(n, p)] = coef.get((n, p), 0.0) + float(c)
        coef[(0, p)] = coef.get((0, p), 0.0) - float(c) * float((x ** n).mean())
    for p, (re, im) in m.b.ell.physical().items():                                       # ell(xi) = sum C_p xi^p  → coefficient of (i xi)^p = C_p / i^p
        cpl = complex(re, im) / (1j ** p)
        if float(p).is_integer() and abs(cpl) > 0:
            coef[(0, int(p))] = coef.get((0, int(p)), 0.0) + cpl.real
TRUE = {1: {(2, 0): 1.0, (0, 2): -1.0}, 2: {(1, 1): 1.0, (0, 2): 1.0}, 3: {(1, 1): 1.0, (0, 2): 1.0}}[g]
keys = sorted(set(coef) | set(TRUE))
err = {f'x^{n} d^{p}': round(coef.get((n, p), 0.0) - TRUE.get((n, p), 0.0), 6) for (n, p) in keys}
res = dict(g=g, seed=seed, support=[list(ATOMS[i]) for i in Ssel], use_phi=phi_sel, val=va, train=trn, test=te,
           coef={f'x^{n} d^{p}': round(v, 6) for (n, p), v in sorted(coef.items()) if abs(v) > 1e-6},
           coef_err=err, max_abs_coef_err=max(abs(v) for v in err.values()),
           trials={f'{[ATOMS[i] for i in s_]} phi={ph}': v for (s_, ph), (v, _) in trials.items()})
print(json.dumps(res), flush=True)
json.dump(res, open(f'{R}/runs/synth/G{g}_U2sym_s{seed}.json', 'w'))
