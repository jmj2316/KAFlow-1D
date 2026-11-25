"""Validates the evaluation harness by reproducing Table 1 of the KANO paper (KANO rows) with the official KANO weights.
 train 2000 (Group A), test 400 (Group B), metric = mean per-sample relative L2 (official rel_l2_loss)."""
import sys, os, math, json
import torch, torch.nn as nn
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import ext_kano_official as KO
from kan import KAN
dev = torch.device('cpu')
R = os.path.join(ROOT, 'ext', 'KANO')


def rel(pred, gf):
    return float(((pred - gf).norm() / gf.norm()))


def load(g):
    d = {}
    for nm in ('train', 'test', 'sample'):
        d[nm] = torch.load(f'{R}/G_{g}/dataset/{nm}_{g}.pt', weights_only=False, map_location='cpu')
    return d


def build(g):
    if g == 1:
        sym = KAN(width=[2, 2], grid=10, k=3, seed=0, device=dev); act = nn.Identity()
    elif g == 2:
        sym = KAN(width=[[2, 0], [1, 1]], grid=10, k=3, seed=0, base_fun='identity', device=dev); act = nn.Identity()
    else:
        sym = KAN(width=[[2, 0], [1, 1]], grid=10, k=3, seed=10, base_fun='identity', device=dev)
        act = KAN(width=[2, 1], grid=10, k=3, seed=11, base_fun='identity', device=dev)
        act.load_state_dict(torch.load(f'{R}/G_3/models/kanoactkan_3.pth', weights_only=False, map_location='cpu'))
    sym.load_state_dict(torch.load(f'{R}/G_{g}/models/kanosymkan_{g}.pth', weights_only=False, map_location='cpu'))
    return KO.Kano(num_layers=1, symbol_net=sym, act_net=act)


paper = {1: (1.04e-4, 1.44e-4), 2: (0.629e-4, 0.749e-4), 3: (0.716e-4, 0.737e-4)}
for g in (1, 2, 3):
    D = load(g); m = build(g); m.eval()
    out = {}
    with torch.no_grad():
        for nm, nmax in (('train', 200), ('test', 400)):   # train uses the first 200 for speed (all 2000 are slow)
            ls = [rel(m(x, f), gf) for (x, f, gf, _) in D[nm][:nmax]]
            out[nm] = sum(ls) / len(ls)
    print(f'G_{g}: reproduced train {out["train"]:.3e} test {out["test"]:.3e} ratio {out["test"]/out["train"]:.2f} | paper train {paper[g][0]:.3e} test {paper[g][1]:.3e} ratio {paper[g][1]/paper[g][0]:.2f}', flush=True)
