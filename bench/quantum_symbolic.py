"""Automatic symbolic distillation of the potential v(x) of the quantum model → better identification of the Kerr coefficient.
 Rationale: over the training window (10 steps) rho(x,t) is nearly static, so the Kerr term c*rho acts like a per-sample static potential.
       A free-form spline v(x) can absorb its mean component c*<rho>(x), which underestimates Kerr (distribution scenarios).
 Procedure (same rules as for the G operators): sparse approximation of the learned v(x) on the polynomial dictionary {x..x^4} (+constant) → candidates (BIC floor path + by size, collinear sets excluded)
       → fix v to a polynomial and retrain for each candidate → smallest support within 1.1 times the best training-window loss → reselect the Kerr support → float64 L-BFGS.
 python -B quantum_symbolic.py nlse posmom 0
"""
import sys, os, json, math, copy, itertools
import torch
import torch.nn as nn
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'bench'))
torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 2)))
from kaf_q import KAFlowQ

R = ROOT
ds, scen, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
MAXT = 10; S = 200
raw = torch.load(f'{R}/data/quantum/{"doublewell" if ds == "dw" else "nlse"}_100.pt', weights_only=False, map_location='cpu')
psi0 = torch.zeros(S, 128, dtype=torch.complex128); psi = torch.zeros(S, 100, 128, dtype=torch.complex128)
pos = torch.zeros(S, 100, 128); mom = torch.zeros(S, 100, 128)
for a, b, lab, s, t in raw:
    if lab == 'initial': psi0[s] = b.to(torch.complex128)
    elif lab == 'wavefunc': psi[s, t - 1] = b.to(torch.complex128)
    elif lab == 'pos_pdf': pos[s, t - 1] = b.double()
    elif lab == 'mom_pdf': mom[s, t - 1] = b.double()
nrm = lambda z: z / z.abs().pow(2).sum(-1, keepdim=True).sqrt()
psi0 = nrm(psi0); psi = nrm(psi)
# target distributions are stored in float32 and do not sum exactly to 1 (~1e-7) → the KL floor (~1e-10) hides the Kerr signal (~1e-10) → renormalize in float64
pos = pos / pos.sum(-1, keepdim=True); mom = mom / mom.sum(-1, keepdim=True)

# same data statistics as run_quantum3 → same model configuration
seqs = torch.cat([psi0[:, None], psi[:, :MAXT]], 1)
Ek = (torch.fft.fft(seqs).abs() ** 2).mean((0, 1))
kk = (2 * math.pi * torch.fft.fftfreq(128, d=4.0 / 128)).abs()
xr = float(torch.sqrt((Ek * kk ** 2).sum() / Ek.sum()))
rate = float(((psi[:, 1:MAXT] - psi[:, :MAXT - 1]).abs().pow(2).mean().sqrt() / psi[:, :MAXT].abs().pow(2).mean().sqrt()) / 1e-4)
base = KAFlowQ(nonlinear=(ds == 'nlse'), tmode='dictpos', E=Ek, xi_ref=xr, tscale=rate)
base.load_state_dict(torch.load(f'{R}/runs/quantum/{ds}_{scen}_U3_s{seed}.pt', map_location='cpu'))


def kl(t, p, eps=1e-12):
    return (t * (torch.log(t + eps) - torch.log(p + eps))).sum(-1)


def loss_fn(m):
    pr = m(psi0, MAXT)[:, 1:MAXT + 1]
    if scen == 'full':
        return ((pr - psi[:, :MAXT]).abs().pow(2).sum(-1).sqrt() / psi[:, :MAXT].abs().pow(2).sum(-1).sqrt()).mean()
    px = pr.abs().pow(2); px = px / px.sum(-1, keepdim=True); l = kl(pos[:, :MAXT], px).mean()
    if scen == 'posmom':
        pk = torch.fft.fft(pr).abs().pow(2); pk = pk / pk.sum(-1, keepdim=True); l = l + kl(mom[:, :MAXT], pk).mean()
    return l


class PolyV(nn.Module):
    """v(x) = c0 + sum_{n in S} c_n x^n   (the constant = global phase / energy reference, always included)"""

    def __init__(self, Sn, c0):
        super().__init__(); self.Sn = list(Sn); self.c = nn.Parameter(c0.clone())

    def forward(self, x):
        return self.c[0] + sum(self.c[1 + j] * x ** n for j, n in enumerate(self.Sn))


# --- v(x) candidates: least squares weighted by the mean density (identifiable only where the wave function lives)
with torch.no_grad():
    xg = base.x; vx = base.v(xg)
    wr = seqs.abs().pow(2).mean((0, 1)); wr = (wr / wr.max()).sqrt()
POW = [1, 2, 3, 4]
def vfit(Sn):
    A = torch.stack([torch.ones_like(xg)] + [xg ** n for n in Sn], 1) * wr[:, None]
    c = torch.linalg.lstsq(A, (vx * wr)[:, None]).solution[:, 0]
    return c, float(((A @ c) - vx * wr).pow(2).sum()), A
def vcands():
    out = set(); yy = float((vx * wr).pow(2).sum()); n = float((wr ** 2).sum() ** 2 / (wr ** 4).sum())
    for fl in (1e-3, 3e-3, 1e-2, 3e-2):
        best = (float('inf'), ())
        for r in range(1, 5):
            for Sn in itertools.combinations(POW, r):
                c, res, A = vfit(Sn)
                An = A / A.norm(dim=0, keepdim=True); sv = torch.linalg.svdvals(An)
                if float(sv[0] / sv[-1]) > 1e3:
                    continue
                bic = n * math.log(res / yy + fl ** 2) + r * math.log(n)
                if bic < best[0]:
                    best = (bic, Sn)
        out.add(best[1])
    for size in (1, 2, 3):
        best = (float('inf'), ())
        for Sn in itertools.combinations(POW, size):
            c, res, A = vfit(Sn)
            if res < best[0]:
                best = (res, Sn)
        out.add(best[1])
    return [s for s in out if s]


def train(m, adam=500, lbfgs=300, lr=1e-3):
    ps = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(ps, lr=lr)
    for it in range(adam):
        for gp in opt.param_groups:
            gp['lr'] = lr * 0.5 * (1 + math.cos(math.pi * it / adam)) + 1e-6
        l = loss_fn(m); opt.zero_grad(); l.backward(); opt.step()
    lb = torch.optim.LBFGS(ps, lr=1.0, max_iter=lbfgs, history_size=50, line_search_fn='strong_wolfe', tolerance_grad=1e-15, tolerance_change=1e-20)
    def cl():
        lb.zero_grad(); l = loss_fn(m); l.backward(); return l
    lb.step(cl)
    return m


def make(Sn, kerr_keep=(1, 1)):
    m = copy.deepcopy(base)
    c0, _, _ = vfit(Sn)
    m.v = PolyV(Sn, c0)
    if ds == 'nlse':
        km = torch.tensor(kerr_keep, dtype=torch.float64)
        with torch.no_grad():
            m.nc.mul_(km)
        m.nc.register_hook(lambda gr, km=km: gr * km)
    return m


trials = {}
for Sn in vcands():
    m = train(make(Sn))
    with torch.no_grad():
        v = float(loss_fn(m))
    trials[Sn] = (v if v == v else float('inf'), m)
    print(f'  v candidate {Sn}: training-window loss {v:.4e}' + (f' Kerr {[round(float(c), 4) for c in m.nc]}' if ds == 'nlse' else ''), flush=True)
vmin = min(v for v, _ in trials.values())
Vsel = min([s for s, (v, _) in trials.items() if v <= 1.1 * vmin], key=len)
print(f'v selected support {Vsel}', flush=True)

kerr_sel = None
if ds == 'nlse':
    # Kerr is a weakly identified direction with a very flat loss valley → profile likelihood instead of gradient descent:
    #   fix the coefficient on a grid and reoptimize the rest (T, v), giving L(c) → parabolic refinement of the minimum → full fine-tuning.
    import numpy as np
    vbase = trials[Vsel][1]

    def profile(idx):
        grid = list(np.linspace(-0.5, 2.0, 11)); Ls = []
        for c in grid:
            m = copy.deepcopy(vbase)
            with torch.no_grad():
                m.nc.zero_(); m.nc[idx] = float(c)
            m.nc.requires_grad_(False)
            m = train(m, adam=0, lbfgs=200)
            with torch.no_grad():
                Ls.append(float(loss_fn(m)))
        i = int(np.argmin(Ls)); j = min(max(i, 1), len(grid) - 2)
        xs, ys = grid[j - 1:j + 2], Ls[j - 1:j + 2]
        A = np.vstack([np.array(xs) ** 2, xs, np.ones(3)]).T; a2, a1, _ = np.linalg.solve(A, ys)
        cstar = float(-a1 / (2 * a2)) if a2 > 0 else float(grid[i])
        print(f'  Kerr profile idx={idx}: grid minimum c={grid[i]:.2f}, parabolic refinement c*={cstar:.4f}', flush=True)
        return cstar

    kt = {}
    for keep in ((1, 0), (0, 1), (1, 1)):
        m = copy.deepcopy(vbase)
        km = torch.tensor(keep, dtype=torch.float64)
        with torch.no_grad():
            m.nc.zero_()
            if keep == (1, 1):
                m.nc[0] = kt[(1, 0)][2]                                   # the two-term candidate starts from the one-term profile solution
            else:
                m.nc[keep.index(1)] = profile(keep.index(1))
        m.nc.requires_grad_(True); m.nc.register_hook(lambda gr, km=km: gr * km)
        m = train(m, adam=0, lbfgs=500)
        with torch.no_grad():
            kt[keep] = (float(loss_fn(m)), m, float(m.nc[0]))
        print(f'  Kerr support {keep}: loss {kt[keep][0]:.4e} coefficients {[round(float(c), 5) for c in m.nc]}', flush=True)
    kmin = min(v[0] for v in kt.values())
    kerr_sel = min([k for k, v in kt.items() if v[0] <= 1.1 * kmin], key=sum)
    model = kt[kerr_sel][1]
else:
    model = trials[Vsel][1]
model = train(model, adam=0, lbfgs=1000)                              # final polishing

with torch.no_grad():
    traj = model(psi0, 100)[:, 1:]
    F = (nrm(traj).conj() * psi).sum(-1).abs(); infid = (1 - F).mean(0)
    px = traj.abs().pow(2); px = px / px.sum(-1, keepdim=True); bcp = (1 - (px * pos).sqrt().sum(-1)).mean(0)
    Tterms = {str(p): v[0] for p, v in model.Td.physical().items() if abs(v[0]) > 1e-5}
    vc = {('1' if j == 0 else f'x^{model.v.Sn[j - 1]}'): round(float(c), 6) for j, c in enumerate(model.v.c)}
res = dict(ds=ds, scen=scen, seed=seed, v_support=list(Vsel), kerr_support=kerr_sel,
           infid_wf={t: float(infid[t - 1]) for t in (1, 10, 50, 100)}, infid_pos100=float(bcp[-1]),
           T_terms=Tterms, v_coef=vc, kerr=[round(float(c), 6) for c in model.nc] if ds == 'nlse' else None,
           v_trials={str(s): v for s, (v, _) in trials.items()})
print(json.dumps(res), flush=True)
json.dump(res, open(f'{R}/runs/quantum/{ds}_{scen}_U3sym_s{seed}.json', 'w'))
