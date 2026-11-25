"""Reproduction of the KANO quantum benchmark protocol (DW, NLSE) + KAFlow-Q.
 Training: first 10 steps of 200 trajectories (rollout loss over T=1..10). Evaluation: 100 steps from the same initial states, fidelity F=|<psi_hat|psi>| (official wf_fidelity), 1-F.
 Scenarios: full (complex relative L2) / posmom (KL of the position and momentum PMFs) / pos (KL of the position PMF)
 python -B run_quantum.py dw full 0 [spline|power]
"""
import sys, os, json, math, time
import numpy as np
import torch
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'bench'))
torch.set_default_dtype(torch.float64)
from kaf_q import KAFlowQ

dev = torch.device(os.environ.get('DEVICE', 'cpu'))
R = ROOT
ds, scen, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
tmode = sys.argv[4] if len(sys.argv) > 4 else 'spline'
tag = sys.argv[5] if len(sys.argv) > 5 else tmode
MAXT = 10
torch.manual_seed(seed)

raw = torch.load(f'{R}/data/quantum/{"doublewell" if ds == "dw" else "nlse"}_100.pt', weights_only=False, map_location='cpu')
S = 200
psi0 = torch.zeros(S, 128, dtype=torch.complex128); psi = torch.zeros(S, 100, 128, dtype=torch.complex128)
pos = torch.zeros(S, 100, 128); mom = torch.zeros(S, 100, 128)
for a, b, lab, s, t in raw:
    if lab == 'initial':
        psi0[s] = b.to(torch.complex128)
    elif lab == 'wavefunc':
        psi[s, t - 1] = b.to(torch.complex128)
    elif lab == 'pos_pdf':
        pos[s, t - 1] = b.double()
    elif lab == 'mom_pdf':
        mom[s, t - 1] = b.double()
nrm = lambda z: z / z.abs().pow(2).sum(-1, keepdim=True).sqrt()
psi0 = nrm(psi0).to(dev); psi = nrm(psi).to(dev); pos = pos.to(dev); mom = mom.to(dev)

LAM1 = float(os.environ.get('LAM1', 1e-5)); LAMR = float(os.environ.get('LAMR', 1e-3))
# penalty normalization: scaled by the loss of the 'null predictor' (psi_t = psi_0) → penalties in units of the 'change of the dynamics', independent of scenario and loss type
with torch.no_grad():
    _null = psi0[:, None].expand(-1, MAXT + 1, -1)
    def _null_loss():
        pr = _null[:, 1:MAXT + 1]
        if scen == 'full':
            return float(((pr - psi[:, :MAXT]).abs().pow(2).sum(-1).sqrt() / psi[:, :MAXT].abs().pow(2).sum(-1).sqrt()).mean())
        px = pr.abs().pow(2); px = px / px.sum(-1, keepdim=True)
        tgt = pos[:, :MAXT]; l = float((tgt * (torch.log(tgt + 1e-12) - torch.log(px + 1e-12))).sum(-1).mean())
        if scen == 'posmom':
            pk = torch.fft.fft(pr).abs().pow(2); pk = pk / pk.sum(-1, keepdim=True); tm = mom[:, :MAXT]
            l += float((tm * (torch.log(tm + 1e-12) - torch.log(pk + 1e-12))).sum(-1).mean())
        return l
    L_NULL = _null_loss()
LAM1 *= L_NULL; LAMR *= L_NULL
print(f'L_null {L_NULL:.3e} → penalties lam1 {LAM1:.2e} lamr {LAMR:.2e}', flush=True)
qkw = {}
if tmode in ('dict', 'dictpos'):
    seqs = torch.cat([psi0[:, None], psi[:, :MAXT]], 1)                                   # statistics from the training window (t<=10) only
    Ek = (torch.fft.fft(seqs).abs() ** 2).mean((0, 1)).cpu()
    kk = (2 * math.pi * torch.fft.fftfreq(128, d=4.0 / 128)).abs()
    xr = float(torch.sqrt((Ek * kk ** 2).sum() / Ek.sum()))
    rate = float(((psi[:, 1:MAXT] - psi[:, :MAXT - 1]).abs().pow(2).mean().sqrt() / psi[:, :MAXT].abs().pow(2).mean().sqrt()) / 1e-4)
    qkw = dict(E=Ek, xi_ref=xr, tscale=rate)                                               # scale = relative rate of change of the data (preprocessing)
    print(f'xi_ref {xr:.3f} rate {rate:.2f}', flush=True)


def make_model(s_):
    torch.manual_seed(s_)
    return KAFlowQ(nonlinear=(ds == 'nlse'), tmode=tmode, **qkw).to(dev)


model = make_model(seed)
npar = model.n_params()


def kl(target, pred, eps=1e-12):
    return (target * (torch.log(target + eps) - torch.log(pred + eps))).sum(-1)


def loss_fn(traj):                                    # traj: (S, MAXT+1, N), t=0..MAXT
    pr = traj[:, 1:MAXT + 1]
    if scen == 'full':
        return ((pr - psi[:, :MAXT]).abs().pow(2).sum(-1).sqrt() / psi[:, :MAXT].abs().pow(2).sum(-1).sqrt()).mean()
    px = pr.abs().pow(2); px = px / px.sum(-1, keepdim=True)
    l = kl(pos[:, :MAXT], px).mean()
    if scen == 'posmom':
        pk = torch.fft.fft(pr).abs().pow(2); pk = pk / pk.sum(-1, keepdim=True)
        l = l + kl(mom[:, :MAXT], pk).mean()
    return l


t0 = time.time()
if tmode in ('dict', 'dictpos'):
    cands = []
    for c in range(3):
        mc = make_model(seed * 100 + c); oc = torch.optim.Adam(mc.parameters(), lr=1e-2)
        for it in range(1000):
            b = torch.randint(0, S, (64,), device=dev); pr = mc(psi0[b], MAXT)[:, 1:]
            if scen == 'full':
                l = ((pr - psi[b][:, :MAXT]).abs().pow(2).sum(-1).sqrt()).mean()
            else:
                px = pr.abs().pow(2); px = px / px.sum(-1, keepdim=True); l = kl(pos[b][:, :MAXT], px).mean()
                if scen == 'posmom':
                    pk = torch.fft.fft(pr).abs().pow(2); pk = pk / pk.sum(-1, keepdim=True); l = l + kl(mom[b][:, :MAXT], pk).mean()
            l = l + mc.penalty(LAM1, LAMR)
            oc.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(mc.parameters(), 10.0); oc.step()
        with torch.no_grad():
            v = float(loss_fn(mc(psi0, MAXT)))
        cands.append((v if v == v else float('inf'), c, mc)); print(f'candidate {c}: training loss {cands[-1][0]:.3e}', flush=True)
    model = min(cands, key=lambda z: z[0])[2]
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
iters = int(os.environ.get('ITERS', 3000))
for it in range(iters):
    lr = 1e-2 * 0.5 * (1 + math.cos(math.pi * it / iters)) + 1e-5
    for g in opt.param_groups:
        g['lr'] = lr
    b = torch.randint(0, S, (64,), device=dev)
    loss = loss_fn(model(psi0[b], MAXT)) if False else None
    traj = model(psi0[b], MAXT)
    pr = traj[:, 1:MAXT + 1]
    # batch loss (loss_fn uses the full S, so it is indexed for the batch)
    pos_b, mom_b, psi_b = pos[b][:, :MAXT], mom[b][:, :MAXT], psi[b][:, :MAXT]
    if scen == 'full':
        loss = ((pr - psi_b).abs().pow(2).sum(-1).sqrt() / psi_b.abs().pow(2).sum(-1).sqrt()).mean()
    else:
        px = pr.abs().pow(2); px = px / px.sum(-1, keepdim=True); loss = kl(pos_b, px).mean()
        if scen == 'posmom':
            pk = torch.fft.fft(pr).abs().pow(2); pk = pk / pk.sum(-1, keepdim=True); loss = loss + kl(mom_b, pk).mean()
    loss = loss + model.penalty(LAM1, LAMR)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
    if (it + 1) % 500 == 0:
        print(f'it {it+1} loss {loss.item():.4e} t {time.time()-t0:.0f}s', flush=True)

def fix_sign(model, tag=''):
    """Sign convention (kinetic energy >= 0): if the weighted mean of T over the energetic band of the data is negative, H -> -H (T, v and Kerr all flip sign)."""
    with torch.no_grad():
        w = model.Td.Ew / model.Td.Ew.sum()
        if float((w * model.Tk()).sum()) < 0:
            model.Td.c.mul_(-1); model.Td.r.mul_(-1)
            model.v.a.mul_(-1); model.v.b.mul_(-1); model.v.c.mul_(-1)
            if ds == 'nlse':
                model.nc.mul_(-1)
            print(f'sign convention applied{tag} (H -> -H)', flush=True)
            return True
    return False


support = None
sign_flipped = False                                              # dictpos: the T>=0 parameterization has no mirror solution
if tmode in ('dict', 'dictpos'):                                                # T support selection: 4 BIC floor candidates → short fine-tuning → training-window loss, smallest support within 1.1 times
    import copy
    dense_state = copy.deepcopy(model.state_dict()); trials = {}
    for kind, val in (('floor', 1e-3), ('floor', 3e-3), ('floor', 1e-2), ('floor', 3e-2), ('size', 1), ('size', 2), ('size', 3)):
        model.load_state_dict(dense_state); (model.Td.r.requires_grad_(True) if hasattr(model.Td, 'r') else None)
        sup = tuple(model.Td.sparsify(floor_rel=val) if kind == 'floor' else model.Td.sparsify(size=val))
        if sup in trials:
            continue
        oc = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-3)
        for _ in range(500):
            l = loss_fn(model(psi0, MAXT)) + model.penalty(LAM1, LAMR); oc.zero_grad(); l.backward(); oc.step()
        with torch.no_grad():
            trials[sup] = (float(loss_fn(model(psi0, MAXT))), copy.deepcopy(model.state_dict()))
        print(f'  {kind} {val}: T support {list(sup)} → loss {trials[sup][0]:.3e}', flush=True)
    vmin = min(v for v, _ in trials.values())
    support = min([s_ for s_, (v, _) in trials.items() if v <= 1.1 * vmin], key=len)
    model.load_state_dict(trials[support][1]); (model.Td.r.requires_grad_(False) if hasattr(model.Td, 'r') else None); support = list(support)
    print(f'T selected support {support}', flush=True)
    if ds == 'nlse':                                               # Kerr support selection: {rho},{rho^2},{rho,rho^2} → same 1.1-times parsimony rule
        base = copy.deepcopy(model.state_dict()); kt = {}
        for keep in ((1, 0), (0, 1), (1, 1)):
            model.load_state_dict(base)
            km = torch.tensor(keep, dtype=model.nc.dtype, device=dev)
            hook = model.nc.register_hook(lambda g, km=km: g * km)
            with torch.no_grad():
                model.nc.mul_(km)
            oc = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-3)
            for _ in range(500):
                l = loss_fn(model(psi0, MAXT)) + model.penalty(LAM1, LAMR); oc.zero_grad(); l.backward(); oc.step()
            hook.remove()
            with torch.no_grad():
                model.nc.mul_(km)
                kt[keep] = (float(loss_fn(model(psi0, MAXT))), copy.deepcopy(model.state_dict()))
            print(f'  Kerr support {keep}: loss {kt[keep][0]:.3e}, coefficients {[round(float(v), 4) for v in model.nc]}', flush=True)
        kmin = min(v for v, _ in kt.values())
        kkeep = min([k_ for k_, (v, _) in kt.items() if v <= 1.1 * kmin], key=sum)
        model.load_state_dict(kt[kkeep][1]); kmask = torch.tensor(kkeep, dtype=model.nc.dtype, device=dev)
        model.nc.register_hook(lambda g, km=kmask: g * km)
        print(f'Kerr selected support {kkeep}', flush=True)
LB = int(os.environ.get('LB', 400))
if LB > 0:
    ps = [p for p in model.parameters() if p.requires_grad]
    lbo = torch.optim.LBFGS(ps, lr=1.0, max_iter=LB, history_size=50, line_search_fn='strong_wolfe', tolerance_grad=1e-14, tolerance_change=1e-18)

    def closure():
        lbo.zero_grad(); l = loss_fn(model(psi0, MAXT)) + model.penalty(LAM1, LAMR); l.backward(); return l
    l0 = float(closure()); lbo.step(closure)
    print(f'LBFGS: {l0:.4e} -> {float(closure()):.4e}  t {time.time()-t0:.0f}s', flush=True)

# --- evaluation: 100-step rollout, official fidelity
model.eval()
with torch.no_grad():
    traj = model(psi0, 100)[:, 1:]
    F = (nrm(traj).conj() * psi).sum(-1).abs()                       # wf_fidelity per (S,t)
    infid = (1 - F).mean(0)                                          # (100,)
    px = traj.abs().pow(2); px = px / px.sum(-1, keepdim=True)
    pk = torch.fft.fft(traj).abs().pow(2); pk = pk / pk.sum(-1, keepdim=True)
    bc_pos = (1 - (px * pos).sqrt().sum(-1)).mean(0); bc_mom = (1 - (pk * mom).sqrt().sum(-1)).mean(0)
res = dict(ds=ds, scen=scen, seed=seed, tmode=tmode, params=npar, iters=iters, T_support=support, L_null=L_NULL, sign_flipped=sign_flipped,
           infid_wf={t: float(infid[t - 1]) for t in (1, 10, 25, 50, 100)},
           infid_pos={t: float(bc_pos[t - 1]) for t in (10, 100)}, infid_mom={t: float(bc_mom[t - 1]) for t in (10, 100)})

# --- symbolic recovery: T(xi) = c |xi|^alpha ,  polynomial coefficients of v(x), linear coefficient of n(rho)
with torch.no_grad():
    xg = model.x.double(); vx = model.v(xg).cpu().numpy(); xn = xg.cpu().numpy()
    V = np.vstack([xn ** p for p in range(5)]).T
    coef = np.linalg.lstsq(V, vx, rcond=None)[0]
    res['v_poly_x0_to_x4'] = [round(float(c), 4) for c in coef]
    kk = model.k.abs().cpu().numpy(); Tk = model.Tk().cpu().numpy(); ok = kk > 1
    A = np.vstack([np.log(kk[ok]), np.ones(ok.sum())]).T
    a_, c_ = np.linalg.lstsq(A, np.log(np.abs(Tk[ok]) + 1e-30), rcond=None)[0]
    res['T_fit_alpha_c'] = [round(float(a_), 4), round(float(np.exp(c_)), 4)]
    if tmode in ('dict', 'dictpos'):
        res['T_dict_terms'] = {str(p): v for p, v in model.Td.physical(qkw['xi_ref']).items() if abs(v[0]) > 1e-4}
        if ds == 'nlse':
            res['kerr_c1_c2'] = [round(float(v), 4) for v in model.nc]
    if ds == 'nlse' and tmode not in ('dict', 'dictpos'):
        rho = torch.linspace(0.0, 0.3, 61, dtype=torch.float64, device=dev)
        nv = model.n(rho).cpu().numpy(); r = rho.cpu().numpy()
        res['n_rho_slope_const'] = [round(float(v), 4) for v in np.polyfit(r, nv, 1)]
print(json.dumps(res), flush=True)
os.makedirs(f'{R}/runs/quantum', exist_ok=True)
json.dump(res, open(f'{R}/runs/quantum/{ds}_{scen}_{tag}_s{seed}.json', 'w'))
torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f'{R}/runs/quantum/{ds}_{scen}_{tag}_s{seed}.pt')
