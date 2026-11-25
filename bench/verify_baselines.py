"""Reproduces the baseline rows of Table 1 of the KANO paper (FNO/U-FNO/AM-FNO/PDNO) with the official weights. Metric = mean per-sample relative L2 (official)."""
import sys, torch
import os
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'bench'))
import ext_baselines as B
from neuralop.models import FNO
from neuralop.layers.embeddings import GridEmbeddingND
R = os.path.join(ROOT, 'ext', 'KANO')
dev = torch.device('cpu')


def data(g, nm):
    d = torch.load(f'{R}/G_{g}/dataset/{nm}_{g}.pt', weights_only=False, map_location='cpu')
    return torch.stack([e[1] for e in d]), torch.stack([e[2] for e in d])


def build(name, g):
    if name == 'FNO':
        pe = GridEmbeddingND(in_channels=1, dim=1, grid_boundaries=[[0, 1]])      # the weights were trained with the default neuralop grid [0,1] (unlike the [-pi,pi] written in the notebook; found by search)
        m = FNO(n_modes=(128,), hidden_channels=64, in_channels=1, out_channels=1, positional_embedding=pe, n_layers=2,
                fno_skip='linear', channel_mlp_skip='soft-gating')
        w = f'fno_{g}.pth'
    elif name == 'UFNO':
        m = B.UFNO1d(n_modes=128, width=32, in_channels=1, out_channels=1); w = f'ufno1d_G{g}.pth'
    elif name == 'AMFNO':
        m = B.AMFNO1d(width=48, n_basis=12, padding=0, in_channels=1, out_channels=1, mlp_dropout=0.0); w = f'amfno_G{g}.pth'
    else:
        m = B.PDNO1d(width=64, symbol_act='gelu', num_layer=2, num_hidden=20, net_out=True, k_max=None, input_dim=1, output_dim=1); w = f'pdno_g{g}.pth'
    sd = torch.load(f'{R}/G_{g}/models/{w}', weights_only=False, map_location='cpu')
    sd = {k: v for k, v in sd.items() if k != '_metadata'}
    r = m.load_state_dict(sd, strict=False)
    assert not (r.missing_keys or r.unexpected_keys), (name, g, r)
    return m.eval()


def fwd(m, f):
    out = m(f.unsqueeze(1))                       # (B,1,N) format (official convert_dataset_for_fno: f -> (1,n))
    return out.squeeze(1)


paper = {  # (train, test) ×1e-4 , Table 1
    'FNO':  {1: (6.36, 98.8), 2: (10.6, 87.0), 3: (11.4, 81.4)},
}
for g in (1, 2, 3):
    Ftr, Gtr = data(g, 'train'); Fte, Gte = data(g, 'test')
    for name in ('FNO', 'UFNO', 'AMFNO', 'PDNO'):
        try:
            m = build(name, g)
            with torch.no_grad():
                tr = torch.cat([(fwd(m, Ftr[i:i + 250]) - Gtr[i:i + 250]).norm(dim=1) / Gtr[i:i + 250].norm(dim=1) for i in range(0, 2000, 250)]).mean()
                te = (fwd(m, Fte) - Gte).norm(dim=1) / Gte.norm(dim=1)
            npar = sum(p.numel() for p in m.parameters())
            ref = paper.get(name, {}).get(g)
            print(f'G{g} {name:6s} params {npar:>7d} | reproduced train {float(tr)*1e4:7.2f}e-4 test {float(te.mean())*1e4:7.2f}e-4 ratio {float(te.mean()/tr):5.1f}' + (f' | paper {ref[0]:.2f}e-4 / {ref[1]:.1f}e-4' if ref else ''), flush=True)
        except Exception as e:
            print(f'G{g} {name:6s} ERROR: {str(e)[:200]}', flush=True)
