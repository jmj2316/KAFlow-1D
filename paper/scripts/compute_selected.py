"""Numbers for the single models selected by validation loss (no new training; for the dynamics, the FNO runs repeated to record val_full).
 Selection rule (common to all models, test data unused):
   static operators (G1–G3, G4): minimum Group A held-out validation loss (heldout_A)
   quantum: minimum training loss over the training window (10 steps)
   dynamics: minimum full-horizon rollout loss on the last 32 training trajectories (KAFlow val_sparse, FNO val_full)
 python -B paper/scripts/compute_selected.py [G|Q|H|D|all]
 → paper/data/selected.json (updated part by part)
"""
import sys, os, json, math, glob
import numpy as np
import torch
import torch.nn as nn
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
K = ROOT; RK = f'{K}/ext/KANO'
sys.path.insert(0, K); sys.path.insert(0, f'{K}/bench')
torch.set_num_threads(8)
J = lambda p: json.load(open(p))
ld = lambda p: torch.load(p, weights_only=False, map_location='cpu')
OUT = f'{K}/paper/data/selected.json'
S = J(OUT) if os.path.exists(OUT) else {}
part = sys.argv[1] if len(sys.argv) > 1 else 'all'
rel = lambda P, G: ((P - G).norm(dim=1) / G.norm(dim=1))


def argmin(vals):
    v = [x if x == x else float('inf') for x in vals]
    return int(np.argmin(v))


# ======================= G1–G3 =======================
if part in ('G', 'all'):
    import ext_kano_official as KO
    from kan import KAN
    from neuralop.models import FNO
    from neuralop.layers.embeddings import GridEmbeddingND
    from kafu import KAFlowUOp2, spectrum_stats

    def gdata(g, nm):
        d = ld(f'{RK}/G_{g}/dataset/{nm}_{g}.pt')
        return d[0][0].float(), torch.stack([e[1] for e in d]).float(), torch.stack([e[2] for e in d]).float(), [str(e[3]) for e in d]

    def kano_official(g):
        if g == 1:
            sym = KAN(width=[2, 2], grid=10, k=3, seed=0, device='cpu', auto_save=False); act = nn.Identity()
        elif g == 2:
            sym = KAN(width=[[2, 0], [1, 1]], grid=10, k=3, seed=0, base_fun='identity', device='cpu', auto_save=False); act = nn.Identity()
        else:
            sym = KAN(width=[[2, 0], [1, 1]], grid=10, k=3, seed=10, base_fun='identity', device='cpu', auto_save=False)
            act = KAN(width=[2, 1], grid=10, k=3, seed=11, base_fun='identity', device='cpu', auto_save=False)
            act.load_state_dict(ld(f'{RK}/G_3/models/kanoactkan_3.pth'))
        sym.load_state_dict(ld(f'{RK}/G_{g}/models/kanosymkan_{g}.pth'))
        return KO.Kano(num_layers=1, symbol_net=sym, act_net=act).eval()

    def kano_apply(model, x, F):
        lay = model.layers[0]; pdo = lay.pdo
        pdo.build_cache_for_x(x); N = x.shape[0]
        Xg = pdo.x_cache.view(N, 1).expand(N, N); Kg = pdo.k_cache.view(1, N).expand(N, N)
        a = pdo.a_net_2d(torch.stack([Xg.flatten(), Kg.flatten()], 1)); A = (a[:, 0] + 1j * a[:, 1]).view(N, N)
        out = []
        for i in range(0, F.shape[0], 50):
            t = torch.einsum('nyk,by->bnk', pdo.exp_factor_cache, F[i:i + 50].to(torch.complex64))
            gr = (pdo.factor * (A[None] * t).sum(-1)).real
            out.append(gr if isinstance(lay.act_net, nn.Identity) else lay.act_net(torch.stack([gr, F[i:i + 50]], -1).reshape(-1, 2)).reshape(gr.shape))
        return torch.cat(out)

    def fno_official(g):
        pe = GridEmbeddingND(in_channels=1, dim=1, grid_boundaries=[[0, 1]])
        m = FNO(n_modes=(128,), hidden_channels=64, in_channels=1, out_channels=1, positional_embedding=pe, n_layers=2, fno_skip='linear', channel_mlp_skip='soft-gating')
        m.load_state_dict({k: v for k, v in ld(f'{RK}/G_{g}/models/fno_{g}.pth').items() if k != '_metadata'}, strict=False)
        return m.eval()

    def kaflow(g, s):
        _, Ftr, Gtr, _ = gdata(g, 'train')
        E, xr = spectrum_stats(Ftr, 2 * math.pi); sc = float(Gtr.pow(2).mean().sqrt() / Ftr.pow(2).mean().sqrt())
        m = KAFlowUOp2(128, 2 * math.pi, E, xr, sc, R=2, J=1, H=2); m.load_state_dict(ld(f'{K}/runs/synth/G{g}_U2opd_s{s}.pt'))
        return m.eval()

    def closed(c, F, x):
        F = F.double(); xx = x.double(); fh = torch.fft.rfft(F, dim=-1)
        xi = torch.fft.rfftfreq(128, d=2 * math.pi / 128).double() * 2 * math.pi
        D = lambda p: torch.fft.irfft((1j * xi) ** p * fh, n=128, dim=-1) if p else F
        out = torch.zeros_like(F)
        for key, v in c.items():
            if key.startswith('x^'):
                n, p = int(key[2]), int(key.split('d^')[1]); out += v * xx ** n * D(p)
            else:
                fx, fxx = D(1), D(2)
                out += v * {'f^2': F ** 2, 'f^3': F ** 3, 'f f_x': F * fx, 'f^2 f_x': F ** 2 * fx, 'f f_xx': F * fxx, 'f_x^2': fx ** 2,
                            'x f^2': xx * F ** 2, 'x f^3': xx * F ** 3}[key]
        return out

    TRUEG = {1: lambda f, fx, fxx, x: x ** 2 * f - fxx, 2: lambda f, fx, fxx, x: x * fx + fxx, 3: lambda f, fx, fxx, x: f ** 3 + x * fx + fxx}

    def true_op(g, x, F):
        n = F.shape[-1]; dx = (x[-1] - x[0] + (x[1] - x[0])) / n
        k = torch.fft.fftfreq(n, d=float(dx)) * (2 * math.pi); fh = torch.fft.fft(F, dim=-1)
        fx = torch.fft.ifft(1j * k * fh, dim=-1).real; fxx = torch.fft.ifft(-(k ** 2) * fh, dim=-1).real
        return TRUEG[g](F, fx, fxx, x)

    G = {}
    for g in (1, 2, 3):
        r = [J(f'{K}/runs/synth/G{g}_U2opd_s{s}.json') for s in range(6)]
        s = argmin([d['heldout_A'] for d in r]); d = r[s]
        u = [J(f'{K}/runs/synth/G{g}_U_s{t}.json') for t in range(6)]; su = argmin([e['heldout_A'] for e in u])
        if g < 3:
            sy = J(f'{K}/runs/synth/G{g}_U2sym_s{s}.json'); symA, symB = sy['train'], sy['test']
        else:
            sy = J(f'{K}/runs/synth/G3_U2symnl_s{s}.json')
            x, Ftr, Gtr, _ = gdata(3, 'train'); _, Fte, Gte, _ = gdata(3, 'test')
            symA = float(rel(closed(sy['coef'], Ftr, x), Gtr.double()).mean()); symB = float(rel(closed(sy['coef'], Fte, x), Gte.double()).mean())
        G[g] = dict(seed=s, A=d['train'], B=d['test'], params=d['params'], symA=symA, symB=symB, sym_coef=sy['coef'],
                    sym_maxerr=sy['max_abs_coef_err'], sym_support=sy.get('support', sy.get('nl_support')),
                    wo_sep=dict(seed=su, A=u[su]['train'], B=u[su]['test'], params=u[su]['params']))
        x, Fa, Ga, na = gdata(g, 'sample'); _, Fb, Gb, nb = gdata(g, 'test')
        kano, fno, kaf = kano_official(g), fno_official(g), kaflow(g, s)
        with torch.no_grad():
            preds = {'KAFlow': lambda F: kaf(F), 'KANO': lambda F: kano_apply(kano, x, F), 'FNO': lambda F: fno(F.unsqueeze(1)).squeeze(1)}
            fam = {}
            for nm, fn in preds.items():
                dd = {}
                for n_, v in list(zip(na, rel(fn(Fa), Ga).tolist())) + list(zip(nb, rel(fn(Fb), Gb).tolist())):
                    dd.setdefault(n_, []).append(v)
                fam[nm] = {k: sum(v) / len(v) for k, v in dd.items()}
            al = np.linspace(0, 1, 101); cur = {nm: [] for nm in preds}
            for a in al:
                Fm = (1 - a) * Fa + a * Fb; Gm = true_op(g, x, Fm)
                for nm, fn in preds.items():
                    cur[nm].append(float(rel(fn(Fm), Gm).mean()))
            G[g]['family'] = fam; G[g]['interp'] = {nm: [v / c[0] for v in c] for nm, c in cur.items()}
            m = kaf; xg = m.x.double(); xi = m.xi.double()
            P = m.ell()[0].to(torch.complex128)[None, :] + (m.ar_scale * m.ar().double()[:, :, None] * m.mr().to(torch.complex128)[:, None, :]).sum(0)
            i0 = int(torch.argmin(xg.abs())); j = int(torch.argmin((xi - 10).abs()))
            sl = dict(x=xg.tolist(), xi=xi.tolist(), p_xi0_re=P[:, 0].real.tolist(), p_x0_re=P[i0].real.tolist(), p_xij_im=(P[:, j].imag / xi[j]).tolist())
            if g == 3:
                fh = torch.fft.rfft(Fb, dim=-1)
                feats = torch.cat([Fb.unsqueeze(-1), torch.fft.irfft(m.M()[None] * fh[:, None, :], n=128, dim=-1).transpose(1, 2)], -1)
                ph = m.phi(feats, m.x); sel = torch.randperm(Fb.numel(), generator=torch.Generator().manual_seed(0))[:3000]
                sl['phi_f'] = Fb.flatten()[sel].tolist(); sl['phi_val'] = ph.flatten()[sel].tolist()
            G[g]['slices'] = sl
        print('G', g, 'seed', s, f"B {d['test']:.3e} symB {symB:.3e} wo_sep seed {su} B {u[su]['test']:.3e}", flush=True)
    S['G'] = G

# ======================= quantum =======================
if part in ('Q', 'all'):
    torch.set_default_dtype(torch.float64)
    from kaf_q import KAFlowQ
    nrm = lambda z: z / z.abs().pow(2).sum(-1, keepdim=True).sqrt()
    kl = lambda t, p: (t * (torch.log(t + 1e-12) - torch.log(p + 1e-12))).sum(-1)
    Q = {}
    for ds in ('dw', 'nlse'):
        raw = ld(f'{K}/data/quantum/{"doublewell" if ds == "dw" else "nlse"}_100.pt')
        psi0 = torch.zeros(200, 128, dtype=torch.complex128); psi = torch.zeros(200, 100, 128, dtype=torch.complex128)
        pos = torch.zeros(200, 100, 128); mom = torch.zeros(200, 100, 128)
        for a, b, lab, s, t in raw:
            if lab == 'initial': psi0[s] = b.to(torch.complex128)
            elif lab == 'wavefunc': psi[s, t - 1] = b.to(torch.complex128)
            elif lab == 'pos_pdf': pos[s, t - 1] = b.double()
            elif lab == 'mom_pdf': mom[s, t - 1] = b.double()
        psi0, psi = nrm(psi0), nrm(psi)
        seqs = torch.cat([psi0[:, None], psi[:, :10]], 1)
        Ek = (torch.fft.fft(seqs).abs() ** 2).mean((0, 1)); kk = (2 * math.pi * torch.fft.fftfreq(128, d=4.0 / 128)).abs()
        xr = float(torch.sqrt((Ek * kk ** 2).sum() / Ek.sum()))
        rate = float(((psi[:, 1:10] - psi[:, :9]).abs().pow(2).mean().sqrt() / psi[:, :10].abs().pow(2).mean().sqrt()) / 1e-4)
        for scen in ('full', 'posmom', 'pos'):
            losses, curves = [], []
            for s in range(3):
                m = KAFlowQ(nonlinear=(ds == 'nlse'), tmode='dictpos', E=Ek, xi_ref=xr, tscale=rate)
                m.load_state_dict(ld(f'{K}/runs/quantum/{ds}_{scen}_U3_s{s}.pt')); m.eval()
                with torch.no_grad():
                    tr = m(psi0, 100)[:, 1:]; pr = tr[:, :10]
                    if scen == 'full':
                        l = float(((pr - psi[:, :10]).abs().pow(2).sum(-1).sqrt() / psi[:, :10].abs().pow(2).sum(-1).sqrt()).mean())
                    else:
                        px = pr.abs().pow(2); px = px / px.sum(-1, keepdim=True); l = float(kl(pos[:, :10], px).mean())
                        if scen == 'posmom':
                            pk = torch.fft.fft(pr).abs().pow(2); pk = pk / pk.sum(-1, keepdim=True); l += float(kl(mom[:, :10], pk).mean())
                    losses.append(l); curves.append((1 - (nrm(tr).conj() * psi).sum(-1).abs()).mean(0).tolist())
            s = argmin(losses); nj = J(f'{K}/runs/quantum/{ds}_{scen}_U3_s{s}.json')
            symf = f'{K}/runs/quantum/{ds}_{scen}_U3sym_s{s}.json'
            sy = J(symf) if os.path.exists(symf) else None
            Q[f'{ds}_{scen}'] = dict(seed=s, train_losses=losses, final=curves[s][-1], curve=curves[s], v_poly=nj.get('v_poly_x0_to_x4'),
                                     T=nj.get('T_dict_terms'), sym=(dict(T=sy['T_terms'], v=sy['v_coef'], kerr=sy['kerr'], final=sy['infid_wf']['100']) if sy else None))
            print('Q', ds, scen, 'seed', s, f'{curves[s][-1]:.2e}', flush=True)
    torch.set_default_dtype(torch.float32)
    S['Q'] = Q

# ======================= G4 (H1) =======================
if part in ('H', 'all'):
    from kafu import KAFlowUOp2, spectrum_stats
    H = {}
    for nm, f, sub in (('FNO', 'synthH/H1_fno_s{}', None), ('KANO', 'synthH/H1_kano_s{}', 'kano'), ('KANO_mlp', 'synthH/H1_kano_s{}', 'kano_mlp'),
                       ('KAFlow_R2', 'synth/H1_R2d_s{}', None), ('KAFlow_R4', 'synth/H1_R4d_s{}', None)):
        r = [J(f'{K}/runs/{f.format(s)}.json') for s in range(3)]
        if sub: r = [d[sub] for d in r]
        s = argmin([d['heldout_A'] for d in r]); d = r[s]
        H[nm] = dict(seed=s, A=d['train'], B=d['test'], rmse=d['test_rmse'], fam=d['test_by_family'], params=d.get('params'))
    H['FNO']['params'] = J(f'{K}/runs/synthH/H1_fno_s0.json')['params']; H['KANO_mlp']['params'] = 1218
    d = ld(f'{K}/data/synthH/H1/train.pt'); Ftr = torch.stack([e[1] for e in d]); Gtr = torch.stack([e[2] for e in d])
    E, xr = spectrum_stats(Ftr, 2 * math.pi); sc = float(Gtr.pow(2).mean().sqrt() / Ftr.pow(2).mean().sqrt())
    m = KAFlowUOp2(128, 2 * math.pi, E, xr, sc, R=4, J=1, H=2); m.load_state_dict(ld(f'{K}/runs/synth/H1_R4d_s{H["KAFlow_R4"]["seed"]}.pt')); m.eval()
    with torch.no_grad():
        P = m.ell()[0].to(torch.complex128)[None, :] + (m.ar_scale * m.ar().double()[:, :, None] * m.mr().to(torch.complex128)[:, None, :]).sum(0)
        xg, xi = m.x.double(), m.xi.double(); T = xi[None, :] ** (1 + 0.5 * torch.sin(xg[:, None]))
    H['symbol'] = dict(x=xg.tolist(), xi=xi.tolist(), learned=P.real.tolist(), true=T.tolist())
    print('H', {k: (v['seed'], round(v['B'], 5)) for k, v in H.items() if k != 'symbol'}, flush=True)
    S['H'] = H

# ======================= dynamics =======================
if part in ('D', 'all'):
    D = {}
    for eq in ('burgers', 'fburgers', 'bo', 'kdv', 'whitham'):
        U = [J(f'{K}/runs/{eq}_U2_s{s}.json') for s in range(6)]
        F = [J(f'{K}/runs/{eq}_fno_tuned_s{s}.json') for s in range(3)]
        assert all('val_full' in f for f in F), 'FNO rerun not finished'
        su = argmin([d['val_sparse'] for d in U]); sf = argmin([f['val_full'] for f in F])
        Sd = J(f'{K}/runs/sindy/{eq}_n0.json')
        row = dict(seed_kaflow=su, seed_fno=sf, steps=[1, 10, 25, 50, 100], kaflow=U[su]['rel_l2'], fno=F[sf]['rel_l2'],
                   kaflow_params=U[su]['params'], fno_params=F[sf]['params'], kaflow_support=U[su]['support'], kaflow_ell=U[su]['ell_terms'])
        for k in ('pdefind', 'weak'):
            v = Sd.get(k); row[k] = v['rel_l2'] if isinstance(v, dict) and 'rel_l2' in v else None
        D[eq] = row
        print('D', eq, 'KAFlow seed', su, f"{U[su]['rel_l2']['100']:.3e}", '| FNO seed', sf, f"{F[sf]['rel_l2']['100']:.3e}", flush=True)
    S['D'] = D

json.dump(S, open(OUT, 'w'))
print('saved', list(S.keys()))
