"""SINDy baselines (PySINDy 1.7.5): PDE-FIND (strong form; spatial spectral or smoothed FD, projection onto the resolved band + column-normalized STLSQ) + Weak SINDy (weak form).
 Library: {u, u^2, u^3} x {D^0..D^4} (with interactions, 19 terms). Threshold and differentiation method are chosen by a 'validation rollout' (the same principle as KAFlow).
 The selected equation is integrated with pseudo-spectral IFRK4 and evaluated on the same test (clean true trajectories, 100 steps).
 python -B sindy_baseline.py burgers 5      (noise in %; 0 = clean data)
"""
import sys, os, json, math, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
import pysindy as ps

ROOT = os.environ.get('KAFLOW_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = ROOT
eq, lvl = sys.argv[1], int(sys.argv[2])
D = np.load(f'{R}/data/{eq}.npz' if lvl == 0 else f'{R}/data/{eq}_n{lvl}.npz')
N, L, dt = int(D['N']), float(D['L']), float(D['dt_obs'])
trall, te = D['train'].astype(np.float64), D['test'].astype(np.float64)
tr, va = trall[:64], trall[-16:]                       # 64 training trajectories (the first ones; 32 for the weak form), 16 validation trajectories (the last ones = KAFlow validation split)
x = np.arange(N) * L / N
T = trall.shape[1]
TRUTH = {'burgers': {'x0_11': 0.02, 'x0x0_1': -1.0}, 'kdv': {'x0_111': -1.0, 'x0x0_1': -6.0},
         'ks': {'x0_11': -1.0, 'x0_1111': -1.0, 'x0x0_1': -1.0}, 'allen_cahn': {'x0': 1.0, 'x0_11': 0.01, 'x0x0x0': -1.0}}

F = [lambda u: u, lambda u: u * u, lambda u: u ** 3]
FN = [lambda s: s, lambda s: s + s, lambda s: s + s + s]


def parse(name):
    """'x0x0x0_11' → (polynomial degree p, derivative order d):  u^p · D^d u  (for d>0 one x0 is part of D^d u)"""
    if '_' in name:
        pre, der = name.split('_')
        return pre.count('x0') - 1, len(der)
    return name.count('x0'), 0


def make_rhs(names, coef):
    k = np.arange(N // 2 + 1); xi = 2 * np.pi * k / L; ik = 1j * xi
    mask = (k <= N // 3).astype(float)
    lin = np.zeros_like(xi, dtype=complex); nl = []
    for n, c in zip(names, coef):
        if abs(c) < 1e-12:
            continue
        p, d = parse(n)
        if d == 0 and p == 1:
            lin += c
        elif d > 0 and p == 0:
            lin += c * ik ** d
        else:
            nl.append((p, d, c))

    def Nh(uh):
        u = np.fft.irfft(uh, n=N, axis=-1); out = np.zeros_like(u)
        for p, d, c in nl:
            Du = np.fft.irfft(ik ** d * uh, n=N, axis=-1) if d > 0 else 1.0
            out += c * (u ** p) * Du
        return np.fft.rfft(out, axis=-1) * mask
    return lin, Nh, mask


def rollout(names, coef, u0, steps, sub=20):
    lin, Nh, mask = make_rhs(names, coef)
    h = dt / sub; E = np.exp(lin * h / 2); E2 = E * E
    uh = np.fft.rfft(u0, axis=-1) * mask; out = [np.fft.irfft(uh, n=N, axis=-1)]
    with np.errstate(all='ignore'):
        for _ in range(steps):
            for _ in range(sub):
                a = Nh(uh); b = Nh(E * (uh + 0.5 * h * a)); c = Nh(E * uh + 0.5 * h * b); dd = Nh(E2 * uh + h * E * c)
                uh = E2 * uh + h / 6 * (E2 * a + 2 * E * (b + c) + dd)
                if not np.isfinite(uh).all():
                    return None
            out.append(np.fft.irfft(uh, n=N, axis=-1))
    return np.stack(out, 1)


def rel_err(pred, true):
    return np.linalg.norm(pred - true, axis=-1) / np.linalg.norm(true, axis=-1)          # (B,T)


MASK = (np.arange(N // 2 + 1) <= N // 3)
_STRONG = {}


def strong_system(method):
    """Strong-form PDE-FIND regression: every library column and the time derivative are projected onto the resolved band (k<=N/3, matching the 2/3 dealiasing of the data generator),
    then STLSQ after column normalization. method: 'spec' (spectral derivatives in space + central differences in time) / 'sfd' (smoothed finite differences in space and time)"""
    if method in _STRONG:
        return _STRONG[method]
    P = lambda arr: np.fft.irfft(np.fft.rfft(arr, axis=0) * MASK[:, None, None], n=N, axis=0)
    dm = ps.SpectralDerivative if method == 'spec' else ps.SmoothedFiniteDifference
    lib = ps.PDELibrary(library_functions=F, function_names=FN, derivative_order=4, spatial_grid=x, include_bias=False,
                        is_uniform=True, periodic=True, differentiation_method=dm)
    tdiff = ps.FiniteDifference(axis=-2) if method == 'spec' else ps.SmoothedFiniteDifference(axis=-2)
    As, bs = [], []
    for i in range(tr.shape[0]):
        U = tr[i].T[:, :, None]
        Th = P(np.asarray(lib.fit_transform(U)))
        Ut = P(np.asarray(tdiff._differentiate(U, dt)))
        As.append(Th[:, 2:-2].reshape(-1, Th.shape[-1])); bs.append(Ut[:, 2:-2].reshape(-1))
    A = np.concatenate(As); b_ = np.concatenate(bs)
    sc = np.linalg.norm(A, axis=0) / np.sqrt(len(A)); bnorm = np.linalg.norm(b_) / np.sqrt(len(b_))
    _STRONG[method] = (lib.get_feature_names(), A / sc, b_ / bnorm, sc, bnorm)
    return _STRONG[method]


def fit(method, thr):
    if method == 'weak':
        U = [tr[i].T[:, :, None] for i in range(32)]
        tt = np.arange(T) * dt
        XT = np.stack(np.meshgrid(x, tt, indexing='ij'), -1)
        lib = ps.WeakPDELibrary(library_functions=F, function_names=FN, derivative_order=4, spatiotemporal_grid=XT,
                                is_uniform=True, K=200, p=4, include_bias=False, periodic=True)
        m = ps.SINDy(feature_library=lib, optimizer=ps.STLSQ(threshold=thr, alpha=1e-5, normalize_columns=True))
        m.fit(U, t=dt, multiple_trajectories=True)
        return m.get_feature_names(), m.coefficients()[0].copy()
    names, An, bn, sc, bnorm = strong_system(method)
    opt = ps.STLSQ(threshold=thr, alpha=1e-8, normalize_columns=False); opt.fit(An, bn)
    return names, opt.coef_[0] / sc * bnorm


t0 = time.time(); trials = []
for method in ('spec', 'sfd', 'weak'):
    for thr in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1):
        try:
            names, coef = fit(method, thr)
            pr = rollout(names, coef, va[:, 0], T - 1)
            v = float(rel_err(pr, va)[:, 1:].mean()) if pr is not None else float('inf')
        except Exception as ex:
            names, coef, v = None, None, float('inf')
        trials.append(dict(method=method, thr=thr, val=v if v == v else float('inf'), names=names, coef=None if coef is None else coef.tolist()))

res = dict(eq=eq, noise=lvl, time=time.time() - t0)
for fam, ms in (('pdefind', ('spec', 'sfd')), ('weak', ('weak',))):
    cand = [t_ for t_ in trials if t_['method'] in ms and t_['names'] is not None]
    if not cand:
        res[fam] = None; continue
    b = min(cand, key=lambda t_: t_['val'])
    names, coef = b['names'], np.array(b['coef'])
    pr = rollout(names, coef, te[:, 0], te.shape[1] - 1)
    if pr is None:
        rl = {s: float('nan') for s in (1, 10, 25, 50, 100)}
    else:
        e = rel_err(pr, te); rl = {s: float(e[:, s].mean()) for s in (1, 10, 25, 50, 100)}
    terms = {n: round(float(c), 5) for n, c in zip(names, coef) if abs(c) > 1e-10}
    out = dict(method=b['method'], thr=b['thr'], val=b['val'], rel_l2=rl, terms=terms, n_terms=len(terms))
    if eq in TRUTH:
        tru = TRUTH[eq]; found = set(terms); tset = set(tru)
        out['precision'] = len(found & tset) / max(len(found), 1); out['recall'] = len(found & tset) / len(tset)
        out['exact'] = found == tset
        out['coef_rel_err'] = {k: (abs(terms.get(k, 0.0) - v) / abs(v)) for k, v in tru.items()}
    res[fam] = out
os.makedirs(f'{R}/runs/sindy', exist_ok=True)
json.dump(res, open(f'{R}/runs/sindy/{eq}_n{lvl}.json', 'w'))
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'terms'}) for k, v in res.items()}), flush=True)
