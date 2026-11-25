"""Fair FNO tuning + multiple seeds. Selection on the validation split (last 32 training trajectories); the test set is evaluated once at the end.
 python -B fno_tune.py sweep burgers      -> runs/{eq}_fno_sweep.jsonl  (width×modes×lr, iters=4000)
 python -B fno_tune.py final burgers 0    -> runs/{eq}_fno_tuned_s{seed}.json  (as in kano_best: best setting of the sweep)
"""
import sys, os, json, math, time, itertools
import numpy as np
import torch
import torch.nn as nn
import baselines as B

dev = torch.device(os.environ.get('DEVICE', 'cpu'))
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.abspath(__file__)))
R = ROOT


def run(eq, width, modes, lr0, iters, seed, mode):
    torch.manual_seed(seed); np.random.seed(seed)
    D = np.load(f'{R}/data/{eq}.npz')
    trall = torch.from_numpy(D['train']).to(dev); te = torch.from_numpy(D['test']).to(dev)
    if mode == 'val':
        tr, te = trall[:-32], trall[-32:]
    else:
        tr = trall
    model = B.FNO1d(width=width, modes=modes).to(dev)
    nn.init.zeros_(model.proj[-1].weight); nn.init.zeros_(model.proj[-1].bias)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr0)
    sched = [(1, iters // 3), (4, iters // 3), (16, iters - 2 * (iters // 3))]
    it = 0; t0 = time.time()
    for K, n_it in sched:
        for _ in range(n_it):
            lr = lr0 * 0.5 * (1 + math.cos(math.pi * it / iters)) + 1e-5
            for g in opt.param_groups:
                g['lr'] = lr
            b = torch.randint(0, tr.shape[0], (32,), device=dev); s0 = torch.randint(0, tr.shape[1] - K, (32,), device=dev)
            seg = tr[b[:, None], s0[:, None] + torch.arange(K + 1, device=dev)[None]]
            h = seg[:, 0]; loss = 0
            for k in range(1, K + 1):
                h = model(h)
                loss = loss + ((h - seg[:, k]).flatten(1).norm(dim=1) / seg[:, k].flatten(1).norm(dim=1)).mean()
            loss = loss / K
            opt.zero_grad(); loss.backward(); B.clip_grad(model.parameters(), 1.0); opt.step()
            it += 1
    model.eval()
    val = None
    if mode == 'final':                                  # for selection: full-horizon rollout loss on the last 32 training trajectories (same definition as KAFlow val_full)
        va = trall[-32:]
        with torch.no_grad():
            h = va[:, 0]; vo = [h]
            for _ in range(va.shape[1] - 1):
                h = model(h); vo.append(h)
            ve = (torch.stack(vo, 1) - va).flatten(2).norm(dim=2) / va.flatten(2).norm(dim=2)
            val = float(ve[:, 1:].mean())
            val = val if val == val else float('inf')
        if os.environ.get('SAVE_CKPT') == '1':
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f'{R}/runs/{eq}_fno_tuned_s{seed}.pt')
    with torch.no_grad():
        h = te[:, 0]; outs = [h]
        for _ in range(te.shape[1] - 1):
            h = model(h); outs.append(h)
        pred = torch.stack(outs, 1)
    err = (pred - te).flatten(2).norm(dim=2) / te.flatten(2).norm(dim=2)
    rmse = ((pred - te) ** 2).mean((0, 2)).sqrt()
    f = lambda v: float(v) if v == v else float('nan')
    return dict(eq=eq, width=width, modes=modes, lr=lr0, iters=iters, seed=seed, mode=mode, params=npar,
                rel_l2={s: f(err[:, s].mean()) for s in (1, 10, 25, 50, 100)},
                rmse={s: f(rmse[s]) for s in (1, 10, 25, 50, 100)}, data_std=float(te.std()), val_full=val, time=time.time() - t0)


def crit(r):
    v = [r['rel_l2'][k] for k in ('1', '10', '25', '50')]
    return sum(v) / 4 if all(x == x for x in v) else float('inf')


if __name__ == '__main__':
    cmd, eq = sys.argv[1], sys.argv[2]
    if cmd == 'sweep':
        out = f'{R}/runs/{eq}_fno_sweep.jsonl'
        for width, modes, lr in itertools.product((32, 64), (16, 32), (1e-3, 3e-3)):
            r = run(eq, width, modes, lr, 4000, 0, 'val')
            open(out, 'a').write(json.dumps(r) + '\n')
    else:                                               # final
        seed = int(sys.argv[3])
        import re
        sweep_eq = re.sub(r'_n\d+$', '', eq)                       # noisy data reuse the setting tuned on clean data
        rows = [json.loads(l) for l in open(f'{R}/runs/{sweep_eq}_fno_sweep.jsonl')]
        b = min(rows, key=crit)
        r = run(eq, b['width'], b['modes'], b['lr'], 4000, seed, 'final')
        json.dump(r, open(f'{R}/runs/{eq}_fno_tuned_s{seed}.json', 'w'))
        print(json.dumps(r))
