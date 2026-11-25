"""KAFlow-U time-stepping training (the same rules for every equation).
 A) 3 multiple starts: 600 iter each (K=1, first 224 training trajectories) → pick one by the K=4 rollout loss on the validation set (last 32)
 B) continue the selected one on all training trajectories (256) with a K=4 → K=16 curriculum
 C) test evaluation with 100-step rollouts (once) + dictionary coefficients in physical units
 python -B train_u2.py burgers 0 U2
"""
import sys, os, json, math, time
import numpy as np
import torch
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kafu import KAFlowU, spectrum_stats, rate_stat

torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 8)))
dev = torch.device(os.environ.get('DEVICE', 'cpu'))
eq, seed = sys.argv[1], int(sys.argv[2]); tag = sys.argv[3] if len(sys.argv) > 3 else 'u'
LAM1 = float(os.environ.get('LAM1', 1e-5)); LAMR = float(os.environ.get('LAMR', 1e-3))
R = ROOT
D = np.load(f'{R}/data/{eq}.npz')
trall = torch.from_numpy(D['train']).to(dev); te = torch.from_numpy(D['test']).to(dev)
tr, va = trall[:-32], trall[-32:]
N, L, dt = int(D['N']), float(D['L']), float(D['dt_obs'])
E, xi_ref = spectrum_stats(tr.cpu(), L)
RATE = rate_stat(tr.cpu(), dt)
rel = lambda a, b: ((a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1)).mean()
# penalty normalization (single rule): proportional to the one-step loss L_null of the null predictor (u_{t+1}=u_t)
with torch.no_grad():
    L_NULL = float(rel(tr[:, :-1].reshape(-1, N), tr[:, 1:].reshape(-1, N)))
LAM1 *= L_NULL; LAMR *= L_NULL
print(f'L_null {L_NULL:.3e} → lam1 {LAM1:.2e} lamr {LAMR:.2e}', flush=True)


def make(s):
    torch.manual_seed(s)
    return KAFlowU(N, L, dt, E, xi_ref, rate=RATE).to(dev)


def train(m, data, sched, lr0, lr_end=1e-5, snap_every=50):
    """On a non-finite loss or gradient, roll back to the last healthy state (parameters + optimizer) and halve the learning rate (backtracking)."""
    import copy
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr0)
    total = sum(n for _, n in sched); it = 0; scale = 1.0; good = 0; nback = 0
    snap = (copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict()))
    for K, n in sched:
        for _ in range(n):
            lr = scale * (lr_end + (lr0 - lr_end) * 0.5 * (1 + math.cos(math.pi * it / total)))
            for g in opt.param_groups:
                g['lr'] = lr
            b = torch.randint(0, data.shape[0], (32,), device=dev); s0 = torch.randint(0, data.shape[1] - K, (32,), device=dev)
            seg = data[b[:, None], s0[:, None] + torch.arange(K + 1, device=dev)[None]]
            pred = m(seg[:, 0], K)
            loss = sum(rel(pred[:, k], seg[:, k]) for k in range(1, K + 1)) / K + m.penalty(LAM1, LAMR)
            opt.zero_grad()
            ok = bool(torch.isfinite(loss))
            if ok:
                loss.backward()
                ok = all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            if ok:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); good += 1
                if good % snap_every == 0:
                    snap = (copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict()))
            else:
                m.load_state_dict(snap[0]); opt.load_state_dict(snap[1]); scale *= 0.5; nback += 1
            it += 1
    if nback:
        print(f'    (rollbacks {nback}, final lr factor {scale:.3g})', flush=True)
    return m


def val_full(m):
    """Mean relative L2 of rollouts over the whole validation trajectories (t=0 → end), the same criterion as the evaluation target (long rollouts). inf on divergence."""
    with torch.no_grad():
        pr = m(va[:, 0], va.shape[1] - 1)
        e = ((pr - va).flatten(2).norm(dim=2) / va.flatten(2).norm(dim=2))[:, 1:].mean()
    v = float(e)
    return v if v == v else float('inf')


def val_loss(m, K=4):
    with torch.no_grad():
        s0 = torch.arange(0, va.shape[1] - K, 10, device=dev)
        seg = va[:, s0[:, None] + torch.arange(K + 1, device=dev)[None]].reshape(-1, K + 1, N)
        pred = m(seg[:, 0], K)
        v = float(sum(rel(pred[:, k], seg[:, k]) for k in range(1, K + 1)) / K)
    return v if v == v else float('inf')


t0 = time.time()
cands = []
for c in range(3):
    m = train(make(seed * 100 + c), tr, [(1, 600)], 1e-2, lr_end=3e-3)
    cands.append((val_full(m), val_loss(m), c, m))
    print(f'candidate {c}: validation (full) {cands[-1][0]:.4e} validation (K=4) {cands[-1][1]:.4e}  t {time.time()-t0:.0f}s', flush=True)
key = (lambda z: z[0]) if any(z[0] < float('inf') for z in cands) else (lambda z: z[1])
vbest, _, cbest, model = min(cands, key=key)
print(f'selected: candidate {cbest} (validation {vbest:.4e})', flush=True)
train(model, trall, [(4, 600), (16, 600)], 5e-3)
v_dense = val_full(model)
print(f'training done t {time.time()-t0:.0f}s, validation (dense) {v_dense:.4e}', flush=True)
# support selection: candidate supports from 4 BIC noise floors → short fine-tuning of each → validation rollout loss, smallest support within 1.1 times the best (parsimony)
import copy
dense_state = copy.deepcopy(model.state_dict())
trials = {}
for kind, val in (('floor', 1e-3), ('floor', 3e-3), ('floor', 1e-2), ('floor', 3e-2), ('size', 1), ('size', 2), ('size', 3)):      # candidates = BIC floor path + best-subset path per size
    model.load_state_dict(dense_state); model.ell.r.requires_grad_(True)
    sup = tuple(model.ell.sparsify(floor_rel=val) if kind == 'floor' else model.ell.sparsify(size=val))
    if sup in trials:
        continue
    model.train(); train(model, tr, [(4, 300)], 2e-3)
    trials[sup] = (val_full(model), copy.deepcopy(model.state_dict()))           # selection = full validation rollout (same as the evaluation target)
    print(f'  {kind} {val}: support {list(sup)} → validation (full) {trials[sup][0]:.4e}', flush=True)
trials[('dense',)] = (v_dense, copy.deepcopy(dense_state))                       # the dense model is also a candidate (treated as the largest support)
vmin = min(v for v, _ in trials.values())
support = min([s_ for s_, (v, _) in trials.items() if v <= 1.1 * vmin], key=lambda s_: 99 if s_ == ('dense',) else len(s_))
model.load_state_dict(trials[support][1]); model.ell.r.requires_grad_(support == ('dense',))
model.train(); train(model, trall, [(16, 300)], 1e-3)
v_sparse = val_full(model); support = list(support)
print(f'selected support {support} → validation after final fine-tuning {v_sparse:.4e}', flush=True)

model.eval()
with torch.no_grad():
    pred = model(te[:, 0], te.shape[1] - 1)
err = (pred - te).flatten(2).norm(dim=2) / te.flatten(2).norm(dim=2)
rmse = ((pred - te) ** 2).mean((0, 2)).sqrt()
f = lambda v: float(v) if v == v else float('nan')
ellc = model.ell.physical(xi_ref)
res = dict(eq=eq, tag=tag, seed=seed, rate=RATE, L_null=L_NULL, support=support, val_dense=v_dense, val_sparse=v_sparse,
           support_trials={str(list(k_)): v_[0] for k_, v_ in trials.items()}, params=model.n_params(), lam1=LAM1, lamr=LAMR, chosen=cbest,
           val_cands=[z[0] for z in cands], xi_ref=xi_ref,
           rel_l2={s: f(err[:, s].mean()) for s in (1, 10, 25, 50, 100)},
           rmse={s: f(rmse[s]) for s in (1, 10, 25, 50, 100)}, data_std=float(te.std()),
           ell_terms={str(p): v for p, v in ellc.items() if abs(v[0]) + abs(v[1]) > 1e-4})
print(json.dumps(res), flush=True)
torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f'{R}/runs/{eq}_{tag}_s{seed}.pt')
json.dump(res, open(f'{R}/runs/{eq}_{tag}_s{seed}.json', 'w'))
