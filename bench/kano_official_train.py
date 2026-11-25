"""The official KANO training recipe (G_2 notebook) applied unchanged to the high-rank operators H1–H3.
 Stage 1 KANO-MLP : SymbolNet(hidden 32, 2 layers) on raw (x,k), one sample at a time, Adam lr 1e-3, 400 epochs, ReduceLROnPlateau(train, 0.5, patience 10),
                  early stop at train<1e-6, fixed data order (no shuffling), imaginary-part L1 term (detached as in the official code, so no gradient)
 Stage 2 distill  : MLP symbol values at 1000 random (x,k) pairs → KAN [[2,0],[1,1]] (grid 10, k 3, base identity) fit(LBFGS 1000, lamb 2, l1 10, entropy 10, grid update 200)
 Stage 3 KANO     : 100 epochs with the KAN symbol, lr 0.1 (0.1, patience 5, grid update every epoch) → 10 epochs lr 1e-5
 The operator is the same Kohn–Nirenberg sum as the official pdo, computed in O(N^2) (relative difference 6.8e-6 from the official O(N^3); bench/generality_check.py).
 python -B bench/kano_official_train.py H1 0
"""
import sys, os, json, math, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 2)))
import ext_kano_official as KO
from kan import KAN

R = ROOT
op, seed = sys.argv[1], int(sys.argv[2])
EP1 = int(os.environ.get('EP1', 400)); EP2 = int(os.environ.get('EP2', 100)); EP3 = int(os.environ.get('EP3', 10))
OUT = f'{R}/runs/synthH'; os.makedirs(OUT, exist_ok=True)
torch.manual_seed(seed); np.random.seed(seed)
ld = lambda nm: torch.load(f'{R}/data/synthH/{op}/{nm}.pt', weights_only=False, map_location='cpu')
train, sample, test = ld('train'), ld('sample'), ld('test')
x0 = train[0][0]


class FastPdo(KO.pdo):
    """Same output as the official pdo (factor * sum_k sum_m A[n,k] e^{ik(x_n-x_m)} f_m) in O(N^2)."""

    def build_cache_for_x(self, x_center):
        super().build_cache_for_x(x_center)
        k, x = self.k_cache, self.x_cache
        self.exp_factor_cache = None                                              # the N^3 tensor is not used
        self.Pn = torch.exp(1j * k[None, :] * x[:, None])                         # (n,k)
        self.E = torch.exp(-1j * k[:, None] * x[None, :])                         # (k,m)
        N = x.shape[0]
        self.xk = torch.stack([x.view(N, 1).expand(N, N).flatten(), k.view(1, N).expand(N, N).flatten()], 1)

    def forward(self, x_center, f_center):
        N = x_center.shape[0]
        if self.x_cache is None or self.n_center != N or not torch.allclose(x_center, self.x_cache):
            self.build_cache_for_x(x_center)
        a = self.a_net_2d(self.xk)
        A = (a[:, 0] + 1j * a[:, 1]).view(N, N)
        fc = f_center if f_center.dim() == 2 else f_center[None]                       # f: (N,) or (B,N)
        g = (self.factor * ((A * self.Pn) @ (self.E @ fc.to(torch.complex64).T))).T
        g = g.squeeze(0) if f_center.dim() == 1 else g
        self.last_imag = g.imag.detach()
        return g.real


def make_kano(sym):
    m = KO.Kano(num_layers=1, symbol_net=sym, act_net=nn.Identity())
    lay = m.layers[0]; fp = FastPdo(sym); lay.pdo = fp
    return m


def rel(p, t):
    return torch.norm(p - t) / (torch.norm(t) + 1e-8)


def train_kano(dset, kano, num_epochs, lr, symbol_is_kan, factor, patience, min_lr=1e-10, early=1e-6, tag=''):
    opt = optim.Adam(kano.parameters(), lr=lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=factor, patience=patience, threshold=1e-4, cooldown=0, min_lr=min_lr)
    kano.layers[0].pdo.build_cache_for_x(dset[0][0])
    t0 = time.time()
    for ep in range(num_epochs):
        if symbol_is_kan:                                                         # update_symbol_net_grid: the full (x,k) grid of the first sample
            sym = kano.layers[0].pdo.a_net_2d
            with torch.no_grad():
                sym.save_act = True; sym.update_grid(kano.layers[0].pdo.xk); sym.save_act = False
        tot = 0.0
        for (x, f, Gf, _) in dset:
            opt.zero_grad()
            pred = kano(x, f)
            l = rel(pred, Gf)
            (l + kano.layers[0].pdo.last_imag.abs().sum()).backward()             # imaginary-part term: detached as in the official code → zero gradient
            opt.step(); tot += l.item()
        tot /= len(dset); sch.step(tot)
        if ep % 10 == 0 or ep == num_epochs - 1:
            print(f'[{tag}] epoch {ep + 1}/{num_epochs} train {tot:.4e} lr {opt.param_groups[0]["lr"]:.1e} ({time.time() - t0:.0f}s)', flush=True)
        if tot < early:
            print(f'[{tag}] early stop at epoch {ep + 1}', flush=True); break
    return kano


def evaluate(kano):
    out = {}
    with torch.no_grad():
        for nm, ds in (('train', train), ('heldout_A', sample), ('test', test)):
            F = torch.stack([e[1] for e in ds]); G = torch.stack([e[2] for e in ds])
            P = kano(x0, F)
            r = (P - G).norm(dim=1) / G.norm(dim=1)
            out[nm] = float(r.mean())
            if nm == 'test':
                out['test_rmse'] = float((P - G).pow(2).mean().sqrt())
                fam = {}
                for e, v in zip(ds, r.tolist()):
                    fam.setdefault(e[3], []).append(v)
                out['test_by_family'] = {k: sum(v) / len(v) for k, v in fam.items()}
    return out


T0 = time.time(); res = dict(op=op, seed=seed)
# 1) KANO-MLP
sym_mlp = KO.SymbolNet(hidden_dim=32, num_layers=2)
kano_mlp = train_kano(train, make_kano(sym_mlp), EP1, 1e-3, False, 0.5, 10, tag='MLP')
kano_mlp.eval(); res['kano_mlp'] = evaluate(kano_mlp); res['kano_mlp']['time'] = time.time() - T0
print('KANO-MLP', json.dumps(res['kano_mlp']), flush=True)
torch.save(sym_mlp.state_dict(), f'{OUT}/{op}_kanosymmlp_s{seed}.pth')

# 2) distillation: MLP symbol -> KAN [[2,0],[1,1]]
xv = np.linspace(-math.pi, math.pi, 1000); kv = np.linspace(-64, 64, 1000)
np.random.shuffle(xv); np.random.shuffle(kv)
xk = torch.tensor(np.column_stack((xv, kv)), dtype=torch.float32)
with torch.no_grad():
    lab = sym_mlp(xk)
idx = np.random.permutation(1000); tr_i, te_i = idx[:900], idx[900:]
dset = {'train_input': xk[tr_i], 'test_input': xk[te_i], 'train_label': lab[tr_i], 'test_label': lab[te_i]}
sym_kan = KAN(width=[[2, 0], [1, 1]], grid=10, k=3, seed=seed, base_fun='identity', device='cpu', auto_save=False)
sym_kan.fit(dset, opt='LBFGS', steps=1000, lamb=2, lamb_l1=10., lamb_entropy=10., update_grid=True, grid_update_num=200,
            start_grid_update_step=-1, stop_grid_update_step=1000, log=100)
sym_kan.save_act = False

# 3) KANO (KAN symbol) fine-tuning
kano = make_kano(sym_kan)
kano = train_kano(train, kano, EP2, 1e-1, True, 0.1, 5, tag='KAN')
kano = train_kano(train, kano, EP3, 1e-5, True, 0.1, 5, tag='KAN2')
kano.eval(); res['kano'] = evaluate(kano); res['kano']['time'] = time.time() - T0
print('KANO', json.dumps(res['kano']), flush=True)
torch.save(sym_kan.state_dict(), f'{OUT}/{op}_kanosymkan_s{seed}.pth')
json.dump(res, open(f'{OUT}/{op}_kano_s{seed}.json', 'w'), indent=1)
