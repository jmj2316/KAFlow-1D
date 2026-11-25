"""The FNO recipe of the official KANO repository (FNO_G_2 notebook) applied unchanged to the high-rank operators H1–H3.
 FNO(n_modes 128, hidden 64, 2 layers, grid embedding [0,1], fno_skip linear, channel_mlp_skip soft-gating): the setting that reproduces the Table 1 FNO row with the official weights
 batch 16 shuffled, Adam lr 1e-3, 1000 epochs, ReduceLROnPlateau(train, 0.5, patience 30, min 1e-10), loss = relative L2 over the whole batch
 python -B bench/fno_official_train.py H1 0
"""
import sys, os, json, time
import torch
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 2)))
from neuralop.models import FNO
from neuralop.layers.embeddings import GridEmbeddingND

R = ROOT
op, seed = sys.argv[1], int(sys.argv[2]); EP = int(os.environ.get('EP', 1000))
OUT = f'{R}/runs/synthH'; os.makedirs(OUT, exist_ok=True)
torch.manual_seed(seed)
ld = lambda nm: torch.load(f'{R}/data/synthH/{op}/{nm}.pt', weights_only=False, map_location='cpu')
train, sample, test = ld('train'), ld('sample'), ld('test')
st = lambda ds, i: torch.stack([e[i] for e in ds]).unsqueeze(1)
Ftr, Gtr = st(train, 1), st(train, 2)
pe = GridEmbeddingND(in_channels=1, dim=1, grid_boundaries=[[0, 1]])
m = FNO(n_modes=(128,), hidden_channels=64, in_channels=1, out_channels=1, positional_embedding=pe, n_layers=2,
        fno_skip='linear', channel_mlp_skip='soft-gating')
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=30, threshold=1e-4, min_lr=1e-10)
rel = lambda p, t: torch.norm(p - t) / (torch.norm(t) + 1e-8)
t0 = time.time()
for ep in range(EP):
    perm = torch.randperm(Ftr.shape[0]); tot = 0.0
    for i in range(0, Ftr.shape[0], 16):
        b = perm[i:i + 16]
        opt.zero_grad(); l = rel(m(Ftr[b]), Gtr[b]); l.backward(); opt.step(); tot += l.item() * len(b)
    tot /= Ftr.shape[0]; sch.step(tot)
    if ep % 50 == 0 or ep == EP - 1:
        print(f'epoch {ep + 1}/{EP} train {tot:.4e} lr {opt.param_groups[0]["lr"]:.1e} ({time.time() - t0:.0f}s)', flush=True)
m.eval(); res = dict(op=op, seed=seed, params=sum(p.numel() for p in m.parameters()))
with torch.no_grad():
    for nm, ds in (('train', train), ('heldout_A', sample), ('test', test)):
        F, G = st(ds, 1), st(ds, 2); P = m(F)
        r = ((P - G).flatten(1).norm(dim=1) / G.flatten(1).norm(dim=1))
        res[nm] = float(r.mean())
        if nm == 'test':
            res['test_rmse'] = float((P - G).pow(2).mean().sqrt())
            fam = {}
            for e, v in zip(ds, r.tolist()):
                fam.setdefault(e[3], []).append(v)
            res['test_by_family'] = {k: sum(v) / len(v) for k, v in fam.items()}
res['time'] = time.time() - t0
print(json.dumps(res), flush=True)
json.dump(res, open(f'{OUT}/{op}_fno_s{seed}.json', 'w'), indent=1)
