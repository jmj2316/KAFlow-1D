"""Quantum, H1, dynamics and cost numbers (evaluating stored models and results only) → paper/data/numbers_rest.json
 python -B paper/scripts/compute_numbers2.py
"""
import sys, os, json, math, statistics as st
import numpy as np
import torch
ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
K = ROOT
sys.path.insert(0, K); sys.path.insert(0, f'{K}/bench')
torch.set_num_threads(8)
from kafu import KAFlowUOp2, spectrum_stats
J = lambda p: json.load(open(p))
ld = lambda p: torch.load(p, weights_only=False, map_location='cpu')
num = {}

# ---------------- quantum: 100-step infidelity curves per scenario (neural model U3) + final values of the symbolic model (U3sym)
torch.set_default_dtype(torch.float64)
from kaf_q import KAFlowQ
nrm = lambda z: z / z.abs().pow(2).sum(-1, keepdim=True).sqrt()
num['quantum'] = {}
for ds in ('dw', 'nlse'):
    raw = ld(f'{K}/data/quantum/{"doublewell" if ds == "dw" else "nlse"}_100.pt')
    psi0 = torch.zeros(200, 128, dtype=torch.complex128); psi = torch.zeros(200, 100, 128, dtype=torch.complex128)
    for a, b, lab, s, t in raw:
        if lab == 'initial': psi0[s] = b.to(torch.complex128)
        elif lab == 'wavefunc': psi[s, t - 1] = b.to(torch.complex128)
    psi0, psi = nrm(psi0), nrm(psi)
    seqs = torch.cat([psi0[:, None], psi[:, :10]], 1)
    Ek = (torch.fft.fft(seqs).abs() ** 2).mean((0, 1)); kk = (2 * math.pi * torch.fft.fftfreq(128, d=4.0 / 128)).abs()
    xr = float(torch.sqrt((Ek * kk ** 2).sum() / Ek.sum()))
    rate = float(((psi[:, 1:10] - psi[:, :9]).abs().pow(2).mean().sqrt() / psi[:, :10].abs().pow(2).mean().sqrt()) / 1e-4)
    for scen in ('full', 'posmom', 'pos'):
        curves, sym = [], []
        for s in range(3):
            m = KAFlowQ(nonlinear=(ds == 'nlse'), tmode='dictpos', E=Ek, xi_ref=xr, tscale=rate)
            m.load_state_dict(ld(f'{K}/runs/quantum/{ds}_{scen}_U3_s{s}.pt')); m.eval()
            with torch.no_grad():
                tr = m(psi0, 100)[:, 1:]
                curves.append((1 - (nrm(tr).conj() * psi).sum(-1).abs()).mean(0).tolist())
            fs = f'{K}/runs/quantum/{ds}_{scen}_U3sym_s{s}.json'
            sym.append(J(fs) if os.path.exists(fs) else None)
            nj = J(f'{K}/runs/quantum/{ds}_{scen}_U3_s{s}.json')
            sym[-1] = dict(sym=sym[-1], neural_readout=dict(v=nj.get('v_poly_x0_to_x4'), T=nj.get('T_dict_terms'), kerr=nj.get('kerr', nj.get('kerr_c'))))
        c = np.array(curves)
        num['quantum'][f'{ds}_{scen}'] = dict(curve=c.mean(0).tolist(), final=float(c[:, -1].mean()), final_seeds=c[:, -1].tolist(),
                                            sym_final=(st.mean(d['sym']['infid_wf']['100'] for d in sym) if sym[0]['sym'] else None),
                                            readouts=[dict(neural=d['neural_readout'], sym=(dict(T=d['sym']['T_terms'], v=d['sym']['v_coef'], kerr=d['sym']['kerr']) if d['sym'] else None)) for d in sym])
        print(ds, scen, num['quantum'][f'{ds}_{scen}']['final'], num['quantum'][f'{ds}_{scen}']['sym_final'], flush=True)
torch.set_default_dtype(torch.float32)

# ---------------- H1 variable-order fractional operator
H = {}
for nm, f in (('FNO', 'fno'), ('KANO', 'kano'), ('KANO_mlp', 'kano')):
    r = [J(f'{K}/runs/synthH/H1_{f}_s{s}.json') for s in range(3)]
    if nm == 'KANO': r = [d['kano'] for d in r]
    if nm == 'KANO_mlp': r = [d['kano_mlp'] for d in r]
    H[nm] = dict(A=st.mean(d['train'] for d in r), B=st.mean(d['test'] for d in r), B_std=st.stdev(d['test'] for d in r),
                 rmse=st.mean(d['test_rmse'] for d in r), fam={k: st.mean(d['test_by_family'][k] for d in r) for k in r[0]['test_by_family']})
for Rr in (2, 4, 8, 16):
    r = [J(f'{K}/runs/synth/H1_R{Rr}d_s{s}.json') for s in range(3)]
    H[f'KAFlow_R{Rr}'] = dict(A=st.mean(d['train'] for d in r), B=st.mean(d['test'] for d in r), B_std=st.stdev(d['test'] for d in r),
                             rmse=st.mean(d['test_rmse'] for d in r), params=r[0]['params'],
                             fam={k: st.mean(d['test_by_family'][k] for d in r) for k in r[0]['test_by_family']})
H['FNO']['params'] = J(f'{K}/runs/synthH/H1_fno_s0.json')['params']; H['KANO_mlp']['params'] = 1218
d = ld(f'{K}/data/synthH/H1/train.pt'); Ftr = torch.stack([e[1] for e in d]); Gtr = torch.stack([e[2] for e in d])
E, xr = spectrum_stats(Ftr, 2 * math.pi); sc = float(Gtr.pow(2).mean().sqrt() / Ftr.pow(2).mean().sqrt())
m = KAFlowUOp2(128, 2 * math.pi, E, xr, sc, R=4, J=1, H=2); m.load_state_dict(ld(f'{K}/runs/synth/H1_R4d_s0.pt')); m.eval()
with torch.no_grad():
    P = m.ell()[0].to(torch.complex128)[None, :] + (m.ar_scale * m.ar().double()[:, :, None] * m.mr().to(torch.complex128)[:, None, :]).sum(0)
    xg, xi = m.x.double(), m.xi.double()
    T = xi[None, :] ** (1 + 0.5 * torch.sin(xg[:, None]))
H['symbol'] = dict(x=xg.tolist(), xi=xi.tolist(), learned=P.real.tolist(), true=T.tolist(), E=(E / E.max()).tolist())
num['H1'] = H
print('H1', {k: (v['B'], v.get('rmse')) for k, v in H.items() if k != 'symbol'}, flush=True)

# ---------------- nonlocal and dispersive PDE dynamics
DYN = {}
for eq in ('burgers', 'fburgers', 'bo', 'kdv', 'whitham'):
    U = [J(f'{K}/runs/{eq}_U2_s{s}.json') for s in range(6)]
    ok = [d for d in U if d['rel_l2']['100'] == d['rel_l2']['100']]
    F = [J(f'{K}/runs/{eq}_fno_tuned_s{s}.json') for s in range(3)]
    S = J(f'{K}/runs/sindy/{eq}_n0.json')
    steps = ['1', '10', '25', '50', '100']
    row = dict(steps=[int(t) for t in steps], conv=len(ok),
               kaflow={t: st.mean(d['rel_l2'][t] for d in ok) for t in steps}, kaflow_std100=st.stdev(d['rel_l2']['100'] for d in ok),
               fno={t: st.mean(d['rel_l2'][t] for d in F) for t in steps}, fno_std100=st.stdev(d['rel_l2']['100'] for d in F),
               fno_params=F[0]['params'], kaflow_params=U[0]['params'],
               supports=[d['support'] for d in U], ell=[d['ell_terms'] for d in U])
    for k in ('pdefind', 'weak'):
        v = S.get(k)
        row[k] = {t: (v['rel_l2'].get(t) if isinstance(v, dict) and 'rel_l2' in v else None) for t in steps}
        row[k + '_expr'] = v.get('expr', v.get('terms')) if isinstance(v, dict) else None
    DYN[eq] = row
    print(eq, row['conv'], row['kaflow']['100'], row['fno']['100'], row['pdefind']['100'], row['weak']['100'], flush=True)
num['dyn'] = DYN
num['cost'] = J(f'{K}/runs/generality_check.json')
json.dump(num, open(f'{K}/paper/data/numbers_rest.json', 'w'))
print('saved')
