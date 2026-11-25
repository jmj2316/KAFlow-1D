"""Speed comparison (CPU, same machine and threads): KANO (official per-sample / batched), FNO (official), KAFlow-U (KAFlowUOp2), KAFlow-U symbolic.
 1) inference time on the 400 G2 test samples   2) one training step (batch 64, forward + backward)   3) scaling with the resolution N (batch-16 inference)
"""
import sys, os, time, math, json
import numpy as np
import torch
import torch.nn as nn
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'bench'))
torch.set_num_threads(8)
import ext_kano_official as KO
from kan import KAN
from neuralop.models import FNO
from neuralop.layers.embeddings import GridEmbeddingND
from kafu import KAFlowUOp2, spectrum_stats


def kano_batched(model, x, F):
    """Batched official KanoLayer (same as interp_test.py; numerically verified against the official per-sample results)."""
    lay = model.layers[0]; pdo = lay.pdo
    pdo.build_cache_for_x(x); N = x.shape[0]
    Xg = pdo.x_cache.view(N, 1).expand(N, N); Kg = pdo.k_cache.view(1, N).expand(N, N)
    a = pdo.a_net_2d(torch.stack([Xg.flatten(), Kg.flatten()], 1))
    A = (a[:, 0] + 1j * a[:, 1]).view(N, N)
    out = []
    for i in range(0, F.shape[0], 50):
        t = torch.einsum('nyk,by->bnk', pdo.exp_factor_cache, F[i:i + 50].to(torch.complex64))
        gr = (pdo.factor * (A[None] * t).sum(-1)).real
        out.append(gr if isinstance(lay.act_net, nn.Identity) else lay.act_net(torch.stack([gr, F[i:i + 50]], -1).reshape(-1, 2)).reshape(gr.shape))
    return torch.cat(out)

R = ROOT; RK = f'{R}/ext/KANO'
ld = lambda p: torch.load(p, weights_only=False, map_location='cpu')


def timeit(fn, reps=5):
    fn()                                                  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


g = 2
d = torch.load(f'{RK}/G_{g}/dataset/test_{g}.pt', weights_only=False, map_location='cpu')
x = d[0][0].float(); F = torch.stack([e[1] for e in d]).float()
dtr = torch.load(f'{RK}/G_{g}/dataset/train_{g}.pt', weights_only=False, map_location='cpu')
Ftr = torch.stack([e[1] for e in dtr]).float(); Gtr = torch.stack([e[2] for e in dtr]).float()

sym = KAN(width=[[2, 0], [1, 1]], grid=10, k=3, seed=0, base_fun='identity', device='cpu')
sym.load_state_dict(ld(f'{RK}/G_2/models/kanosymkan_2.pth'))
kano = KO.Kano(num_layers=1, symbol_net=sym, act_net=nn.Identity()).eval()
pe = GridEmbeddingND(in_channels=1, dim=1, grid_boundaries=[[0, 1]])
fno = FNO(n_modes=(128,), hidden_channels=64, in_channels=1, out_channels=1, positional_embedding=pe, n_layers=2,
          fno_skip='linear', channel_mlp_skip='soft-gating')
fno.load_state_dict({k: v for k, v in ld(f'{RK}/G_2/models/fno_2.pth').items() if k != '_metadata'}, strict=False); fno.eval()
E, xi_ref = spectrum_stats(Ftr, 2 * math.pi); scale = float(Gtr.pow(2).mean().sqrt() / Ftr.pow(2).mean().sqrt())
kaf = KAFlowUOp2(128, 2 * math.pi, E, xi_ref, scale); kaf.load_state_dict(ld(f'{R}/runs/synth/G2_U2op_s0.pt')); kaf.eval()
symc = json.load(open(f'{R}/runs/synth/G2_U2sym_s0.json'))['coef']
k_ = torch.fft.rfftfreq(128, d=2 * math.pi / 128) * 2 * math.pi; ik = 1j * k_


def kaflow_symbolic(f):                                   # distilled closed form: c_{1,1} x f_x + c_{0,2} f_xx (+ negligible terms)
    fh = torch.fft.rfft(f, dim=-1)
    fx = torch.fft.irfft(ik * fh, n=128, dim=-1); fxx = torch.fft.irfft(ik * ik * fh, n=128, dim=-1)
    return symc.get('x^1 d^1', 0) * x * fx + symc.get('x^0 d^2', 0) * fxx + symc.get('x^0 d^1', 0) * fx


params = dict(KANO=sum(p.numel() for p in kano.parameters()), FNO=sum(p.numel() for p in fno.parameters()),
              KAFlow_U=sum(p.numel() for p in kaf.parameters()), KAFlow_U_symbolic=len([v for v in symc.values() if abs(v) > 1e-6]))
res = dict(params=params, infer_400={}, train_step_64={}, scaling={})
with torch.no_grad():
    res['infer_400']['KANO (official per-sample loop)'] = timeit(lambda: [kano(x, F[i]) for i in range(F.shape[0])], reps=3)
    res['infer_400']['KANO (batched)'] = timeit(lambda: kano_batched(kano, x, F))
    res['infer_400']['FNO'] = timeit(lambda: fno(F.unsqueeze(1)))
    res['infer_400']['KAFlow-U'] = timeit(lambda: kaf(F))
    res['infer_400']['KAFlow-U symbolic'] = timeit(lambda: kaflow_symbolic(F))


def step(fwd, ps):
    def f():
        bi = torch.randint(0, Ftr.shape[0], (64,))
        out = fwd(Ftr[bi]); loss = ((out - Gtr[bi]).norm(dim=1) / Gtr[bi].norm(dim=1)).mean()
        for p in ps:
            p.grad = None
        loss.backward()
    return f


res['train_step_64']['KANO (official: 64 samples sequentially)'] = timeit(lambda: [step(lambda F_: kano(x, F_[0])[None], list(kano.parameters()))() for _ in range(64)], reps=3)
res['train_step_64']['KANO (batched)'] = timeit(step(lambda F_: kano_batched(kano, x, F_), list(kano.parameters())))
res['train_step_64']['FNO'] = timeit(step(lambda F_: fno(F_.unsqueeze(1)).squeeze(1), list(fno.parameters())))
res['train_step_64']['KAFlow-U'] = timeit(step(lambda F_: kaf(F_), list(kaf.parameters())))

# resolution scaling (batch-16 inference). The same architecture is rebuilt at each N (weights used for timing only).
for N in (64, 128, 256, 512):
    xs = -math.pi + torch.arange(N, dtype=torch.float32) * 2 * math.pi / N
    fN = torch.randn(16, N)
    EN = torch.ones(N // 2 + 1); EN[0] = 1e-3
    kafN = KAFlowUOp2(N, 2 * math.pi, EN, 3.0, 100.0).eval()
    fnoN = FNO(n_modes=(N,), hidden_channels=64, in_channels=1, out_channels=1, positional_embedding=GridEmbeddingND(in_channels=1, dim=1, grid_boundaries=[[0, 1]]), n_layers=2).eval()
    with torch.no_grad():
        row = {'KANO (batched)': timeit(lambda: kano_batched(kano, xs, fN), reps=3), 'FNO': timeit(lambda: fnoN(fN.unsqueeze(1))),
               'KAFlow-U': timeit(lambda: kafN(fN))}
    res['scaling'][N] = row
    print(f'N={N}: ' + ' | '.join(f'{k} {v*1e3:.1f}ms' for k, v in row.items()), flush=True)
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open(f'{R}/runs/speed.json', 'w'), ensure_ascii=False, indent=1)
