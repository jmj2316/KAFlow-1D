"""Automatic symbolic distillation of the nonlinear part Phi (reads the f^3 coefficient of G3). Same rules as the distillation of the linear part (symbolic_op.py).
 1) Sparse approximation of the learned Phi_nl(f, Mf, x) output on a dictionary of nonlinear monomials → candidate supports (BIC floor path + by size, collinear sets excluded)
    Dictionary: f^2, f^3, f f_x, f^2 f_x, f f_xx, f_x^2, x f^2, x f^3   (the true term is not specified)
 2) For each candidate: remove the Phi network and retrain the fully symbolic form G f = ell(D)f + sum c_np (x^n-<x^n>) d^p f + sum d_j m_j(f)
    (+ a 'Phi network kept' candidate for comparison)
 3) Smallest support within 1.1 times the best Group A held-out validation loss → float64 L-BFGS 2000
 python -B symbolic_op_nl.py 3 0
"""
import sys, os, json, math, copy, itertools
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
lin = json.load(open(f'{R}/runs/synth/G{g}_U2sym_s{seed}.json'))                       # distilled linear part (support)
LIN_S = [tuple(a) for a in lin['support']]
rel = lambda p, t: (p - t).norm(dim=1) / t.norm(dim=1)
x = base.x.double(); xi = base.xi.double()
XC = {n: x ** n - (x ** n).mean() for n in range(1, 5)}


def derivs(f):
    fh = torch.fft.rfft(f, dim=-1); k = (1j * base.xi).to(fh.dtype)
    return torch.fft.irfft(k * fh, n=N, dim=-1), torch.fft.irfft(k * k * fh, n=N, dim=-1)


NL = ['f^2', 'f^3', 'f f_x', 'f^2 f_x', 'f f_xx', 'f_x^2', 'x f^2', 'x f^3']


def nl_atoms(f, xx):
    fx, fxx = derivs(f)
    return torch.stack([f ** 2, f ** 3, f * fx, f ** 2 * fx, f * fxx, fx ** 2, xx * f ** 2, xx * f ** 3], 1)   # (B,P,N)


# --- 1) Phi_nl samples and candidates
with torch.no_grad():
    fh = torch.fft.rfft(Ftr, dim=-1)
    feats = torch.cat([Ftr.unsqueeze(-1), torch.fft.irfft(base.M()[None] * fh[:, None, :], n=N, dim=-1).transpose(1, 2)], -1)
    Y = base.phi(feats, base.x).double().reshape(-1)                                     # (B*N,)
    Atom = nl_atoms(Ftr.double(), x).permute(0, 2, 1).reshape(-1, len(NL))                # (B*N,P)
cn = Atom.norm(dim=0); An = Atom / cn; P = len(NL); n_eff = float(Atom.shape[0])


def fit(S):
    As = An[:, list(S)]
    c = torch.linalg.lstsq(As, Y[:, None]).solution[:, 0]
    return c, float((As @ c - Y).pow(2).sum())


def cands():
    out = set(); yy = float(Y.pow(2).sum()); ymax = float(Y.abs().max())
    for fl in (1e-3, 3e-3, 1e-2, 3e-2):
        best = (float('inf'), ())
        for r in range(1, 5):
            for S in itertools.combinations(range(P), r):
                if r > 1:
                    sv = torch.linalg.svdvals(An[:, list(S)])
                    if float(sv[0] / sv[-1].clamp_min(1e-30)) > 1e3:
                        continue
                _, res = fit(S)
                bic = n_eff * math.log(res / yy + (fl * ymax) ** 2 / float(Y.pow(2).mean())) + r * math.log(n_eff)
                if bic < best[0]:
                    best = (bic, S)
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
    return [s for s in out if s]


# initial linear part: least squares of the learned K(x,xi) on the selected support (as in symbolic_op.py)
with torch.no_grad():
    K = (base.ar_scale * base.ar().double()[:, :, None] * base.mr().to(torch.complex128)[:, None, :]).sum(0)
wk = (E.double() / E.max()).sqrt(); ik = 1j * xi
Alin = torch.stack([((XC[n][:, None] * ik[None, :] ** p) * wk[None, :]).reshape(-1) for (n, p) in LIN_S], 1)
lin0 = torch.linalg.lstsq(Alin, (K * wk[None, :]).reshape(-1, 1)).solution[:, 0].real


class Sym(nn.Module):
    """Fully symbolic form: ell(D)f + sum c_np (x^n-<x^n>) d^p f + sum d_j m_j(f)   (nl=None keeps the Phi network for comparison)"""

    def __init__(self, nlS, d0):
        super().__init__()
        self.b = copy.deepcopy(base)
        with torch.no_grad():
            self.b.ell.r.zero_()
        self.b.ell.r.requires_grad_(False)
        for mod in (self.b.mr, self.b.ar):
            for p_ in mod.parameters():
                p_.requires_grad_(False)
        self.nlS = nlS
        if nlS is not None:
            for p_ in list(self.b.phi.parameters()) + list(self.b.M.parameters()):
                p_.requires_grad_(False)
            self.d = nn.Parameter(d0.float())
        self.register_buffer('Xc', torch.stack([XC[n].float() for n, _ in LIN_S]))
        self.Pw = [p for _, p in LIN_S]
        self.c = nn.Parameter(lin0.float().clone())

    def forward(self, f):
        b = self.b
        fh = torch.fft.rfft(f, dim=-1)
        out = torch.fft.irfft(b.ell()[0] * fh, n=N, dim=-1)
        ikx = (1j * b.xi).to(torch.complex128 if f.dtype == torch.float64 else torch.complex64)
        for j, pp in enumerate(self.Pw):
            Dp = torch.fft.irfft((ikx ** pp if pp else torch.ones_like(ikx)) * fh, n=N, dim=-1)
            out = out + self.c[j] * self.Xc[j] * Dp
        if self.nlS is None:
            feats = [f.unsqueeze(-1), torch.fft.irfft(b.M()[None] * fh[:, None, :], n=N, dim=-1).transpose(1, 2)]
            return out + b.phi(torch.cat(feats, -1), b.x)
        At = nl_atoms(f, b.x.to(f.dtype))[:, list(self.nlS), :]
        return out + (self.d[None, :, None] * At).sum(1)


def train(m):
    ps = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(ps, lr=2e-3)
    for it in range(1000):
        for gp in opt.param_groups:
            gp['lr'] = 2e-3 * 0.5 * (1 + math.cos(math.pi * it / 1000)) + 1e-5
        bi = torch.randint(0, Ftr.shape[0], (64,))
        loss = rel(m(Ftr[bi]), Gtr[bi]).mean(); opt.zero_grad(); loss.backward(); opt.step()
    lb = torch.optim.LBFGS(ps, lr=1.0, max_iter=300, history_size=50, line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-15)

    def cl():
        lb.zero_grad(); l = rel(m(Ftr), Gtr).mean(); l.backward(); return l
    lb.step(cl)
    return m


trials = {}
for S in cands() + [None]:
    if S is None:
        m = train(Sym(None, None)); name = 'Phi network kept'
    else:
        c, _ = fit(S); m = train(Sym(S, c / cn[list(S)])); name = str([NL[i] for i in S])
    with torch.no_grad():
        v = float(rel(m(Fva), Gva).mean())
    trials[S] = (v if v == v else float('inf'), m)
    print(f'  nonlinear candidate {name} → validation {v:.3e}', flush=True)
vmin = min(v for v, _ in trials.values())
Ssel = min([s for s, (v, _) in trials.items() if v <= 1.1 * vmin], key=lambda s: 99 if s is None else len(s))
m = trials[Ssel][1]

# float64 polishing
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

with torch.no_grad():
    te = float(rel(m(Fte), Gte).mean()); va = float(rel(m(Fva), Gva).mean())
    coef = {}
    for (n, p), c in zip(LIN_S, m.c.double()):
        coef[f'x^{n} d^{p}'] = coef.get(f'x^{n} d^{p}', 0.0) + float(c)
        coef[f'x^0 d^{p}'] = coef.get(f'x^0 d^{p}', 0.0) - float(c) * float((x ** n).mean())
    for p, (re, im) in m.b.ell.physical().items():
        cpl = complex(re, im) / (1j ** p)
        if float(p).is_integer() and abs(cpl) > 0:
            coef[f'x^0 d^{int(p)}'] = coef.get(f'x^0 d^{int(p)}', 0.0) + cpl.real
    if Ssel is not None:
        for j, i in enumerate(Ssel):
            coef[NL[i]] = float(m.d[j])
TRUE = {3: {'x^1 d^1': 1.0, 'x^0 d^2': 1.0, 'f^3': 1.0}}.get(g, {})
keys = set(coef) | set(TRUE)
err = {k: round(coef.get(k, 0.0) - TRUE.get(k, 0.0), 7) for k in keys}
res = dict(g=g, seed=seed, lin_support=[list(a) for a in LIN_S], nl_support=None if Ssel is None else [NL[i] for i in Ssel],
           val=va, test=te, coef={k: round(v, 7) for k, v in sorted(coef.items()) if abs(v) > 1e-7},
           coef_err=err, max_abs_coef_err=max(abs(v) for v in err.values()),
           trials={('Phi' if s is None else str([NL[i] for i in s])): v for s, (v, _) in trials.items()})
print(json.dumps(res), flush=True)
json.dump(res, open(f'{R}/runs/synth/G{g}_U2symnl_s{seed}.json', 'w'))
