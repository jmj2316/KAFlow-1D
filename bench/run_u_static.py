"""KAFlow-U static operators: KANO synthetic benchmarks G1-G3 (train on Group A → test on Group B). Same structure for every G (J=1, H=2).
 A) 3 multiple starts, 1000 iter each → pick one by the Group A held-out validation (sample) loss
 B) continue the selected one with Adam 7000 iter + full-batch L-BFGS 800
 python -B bench/run_u_static.py 2 0 U
"""
import sys, os, json, math, time
import torch
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from kafu import KAFlowUOp, spectrum_stats

dev = torch.device(os.environ.get('DEVICE', 'cpu'))
R = ROOT
g, seed = int(sys.argv[1]), int(sys.argv[2]); tag = sys.argv[3] if len(sys.argv) > 3 else 'u'
LAM1 = float(os.environ.get('LAM1', 1e-5)); LAMR = float(os.environ.get('LAMR', 1e-3))


def load(nm):
    d = torch.load(f'{R}/ext/KANO/G_{g}/dataset/{nm}_{g}.pt', weights_only=False, map_location='cpu')
    return torch.stack([e[1] for e in d]).to(dev), torch.stack([e[2] for e in d]).to(dev), [str(e[3]) for e in d]


Ftr, Gtr, _ = load('train'); Fte, Gte, nte = load('test'); Fva, Gva, _ = load('sample')
L = 2 * math.pi; N = 128
E, xi_ref = spectrum_stats(Ftr.cpu(), L)
scale = float(Gtr.pow(2).mean().sqrt() / Ftr.pow(2).mean().sqrt())
rel = lambda p, t: (p - t).norm(dim=1) / t.norm(dim=1)


def make(s):
    torch.manual_seed(s)
    return KAFlowUOp(N, L, E, xi_ref, scale, J=1, H=2).to(dev)


def adam(m, iters, lr0, lr_end=1e-5):
    opt = torch.optim.Adam(m.parameters(), lr=lr0)
    for it in range(iters):
        lr = lr_end + (lr0 - lr_end) * 0.5 * (1 + math.cos(math.pi * it / iters))
        for gp in opt.param_groups:
            gp['lr'] = lr
        b = torch.randint(0, Ftr.shape[0], (64,), device=dev)
        loss = rel(m(Ftr[b]), Gtr[b]).mean() + m.penalty(LAM1, LAMR)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


t0 = time.time()
cands = []
for c in range(3):
    m = adam(make(seed * 100 + c), 1000, 1e-2, lr_end=3e-3)
    with torch.no_grad():
        v = float(rel(m(Fva), Gva).mean())
    cands.append((v if v == v else float('inf'), c, m))
    print(f'candidate {c}: validation (heldout A) {cands[-1][0]:.3e}', flush=True)
vbest, cbest, model = min(cands, key=lambda z: z[0])
adam(model, 7000, 5e-3)
with torch.no_grad():
    v_dense = float(rel(model(Fva), Gva).mean())
import copy
dense_state = copy.deepcopy(model.state_dict()); trials = {}
for kind, val in (('floor', 1e-3), ('floor', 3e-3), ('floor', 1e-2), ('floor', 3e-2), ('size', 1), ('size', 2), ('size', 3)):      # candidates = BIC floor path + best subset per size → short fine-tuning → heldout-A validation, smallest support within 1.1 times
    model.load_state_dict(dense_state); model.ell.r.requires_grad_(True)
    sup = tuple(model.ell.sparsify(floor_rel=val) if kind == 'floor' else model.ell.sparsify(size=val))
    if sup in trials:
        continue
    adam(model, 800, 2e-3)
    with torch.no_grad():
        trials[sup] = (float(rel(model(Fva), Gva).mean()), copy.deepcopy(model.state_dict()))
    print(f'  {kind} {val}: support {list(sup)} → validation {trials[sup][0]:.3e}', flush=True)
vmin = min(v for v, _ in trials.values())
support = min([s_ for s_, (v, _) in trials.items() if v <= 1.1 * vmin], key=len)
model.load_state_dict(trials[support][1]); model.ell.r.requires_grad_(False); support = list(support)
print(f'selected support {support} (dense validation {v_dense:.3e})', flush=True)
lb = torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad], lr=1.0, max_iter=800, history_size=50, line_search_fn='strong_wolfe',
                       tolerance_grad=1e-10, tolerance_change=1e-14)

def closure():
    lb.zero_grad(); l = rel(model(Ftr), Gtr).mean() + model.penalty(LAM1, LAMR); l.backward(); return l
lb.step(closure)
model.eval()
with torch.no_grad():
    r_tr = rel(model(Ftr), Gtr); r_te = rel(model(Fte), Gte); r_va = rel(model(Fva), Gva)
fam = {}
for n_, r_ in zip(nte, r_te.tolist()):
    fam.setdefault(n_, []).append(r_)
res = dict(g=g, seed=seed, tag=tag, params=model.n_params(), chosen=cbest, support=support, val_dense=v_dense, val_cands=[z[0] for z in cands],
           train=float(r_tr.mean()), heldout_A=float(r_va.mean()), test=float(r_te.mean()), ratio=float(r_te.mean() / r_tr.mean()),
           test_by_family={k: sum(v) / len(v) for k, v in fam.items()},
           ell_terms={str(p): v for p, v in model.ell.physical(xi_ref).items() if abs(v[0]) + abs(v[1]) > 1e-4}, time=time.time() - t0)
print(json.dumps(res), flush=True)
os.makedirs(f'{R}/runs/synth', exist_ok=True)
json.dump(res, open(f'{R}/runs/synth/G{g}_{tag}_s{seed}.json', 'w'))
torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f'{R}/runs/synth/G{g}_{tag}_s{seed}.pt')
