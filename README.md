# KAFlow: Low-Rank Kolmogorov–Arnold Generator Flows for Interpretable Operator Learning

Code, trained models and result files for

> Minjong Cheon. *Low-Rank Kolmogorov–Arnold Generator Flows for Interpretable Operator Learning.* arXiv:2610.13028, 2026. https://arxiv.org/abs/2610.13028

KAFlow learns the **generator** of a physical operator as a low-rank pseudo-differential symbol built from univariate Kolmogorov–Arnold functions,

$$\mathcal G_\theta u = \mathrm{Op}(\ell_\theta)u + \sum_{r=1}^{R} a_{\theta,r}(x)\,\mathrm{Op}(m_{\theta,r})u + \mathrm{Op}(d_\theta)\,\Phi^{\mathrm{nl}}_\theta\big(u,\mathrm{Op}(\mu_\theta)u,x\big),$$

fixes the gauge of these functions so that the spectral part $\ell$, the position-dependent part $\sum_r a_r m_r$ and the nonlinear part $\Phi^{\mathrm{nl}}$ are unique, integrates dynamics with exact exponential integrators (Lawson RK4; Strang splitting for quantum dynamics), and distills a closed form automatically from fixed libraries with a BIC path and a parsimony rule. The low-rank quantization costs $\mathcal O(R\,m\log m)$ for $m$ grid points.

## Main results (arXiv v1)

- **KANO synthetic operators G1–G3**: 3.4 to 11 times lower out-of-distribution loss than KANO, and every coefficient of the automatically distilled closed form within $10^{-4}$, at the precision floor of the data.
- **Quantum dynamics (double well, NLSE)**: state infidelity up to 6 orders of magnitude lower than Q-KANO, and 30 times lower when trained only on position and momentum measurements.
- **Variable-order fractional operator** $\mathrm{Op}(|\xi|^{1+0.5\sin x})$: 15 times more accurate than existing operator networks.
- **Evolution equations** (Burgers, fractional Burgers, Benjamin–Ono, KdV, Whitham): nonlocal, fractional and non-polynomial dispersion relations recovered where sparse regression fails.

## Installation

The experiments were run with Python 3.10 and the versions in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The SINDy baselines use PySINDy 1.7.5, which we installed in a separate environment (`pip install pysindy==1.7.5`); it is needed only for `bench/sindy_baseline.py`.

## Data

1. **KANO benchmarks and baseline code.** Clone the official KANO repository into `ext/KANO` and regenerate the two modules that hold its model classes. The KANO repository has no license file, so its code is not redistributed here; `tools/extract_kano.py` copies the classes verbatim from its notebooks.
   ```bash
   git clone https://github.com/jinucsb/KANO ext/KANO
   python tools/extract_kano.py
   ```
   This provides the synthetic-operator datasets (`ext/KANO/G_*/dataset`), the official KANO and baseline checkpoints, `ext_kano_official.py` and `bench/ext_baselines.py`.
2. **Quantum datasets.** Download `doublewell_100.pt` and `nlse_100.pt` from the OSF link given in `ext/KANO/DW/dataset_readme.rtf` and `ext/KANO/NLSE/dataset_readme.rtf`, and place them in `data/quantum/`.
3. **Variable-order fractional operator.** `python bench/synthH_data.py` writes `data/synthH/` (inputs from the official KANO generator; `H1` is the operator $\mathcal G_4$ of the paper).
4. **Evolution equations.** `python pde1d.py burgers fburgers bo kdv whitham` writes `data/{eq}.npz` with fixed seeds.

## Reproducing the paper

| Experiment | Script | Paper |
|---|---|---|
| Synthetic operators G1–G3, ablation, symbolic distillation | `experiments/static_operators.sh` | Tables 1–2, Figs. 2–3 |
| Quantum dynamics | `experiments/quantum.sh` | Tables 3–4, Fig. 4 |
| Variable-order fractional operator | `experiments/variable_order.sh` | Table 5, Fig. 5 |
| Evolution equations, FNO and SINDy baselines | `experiments/dynamics.sh` | Table 6, Fig. 6 |
| Reproduction of the published KANO rows, interpolation test, separation rank, cost | `experiments/checks_and_costs.sh` | Table 1 (baseline rows), Fig. 1C, Table 7 |
| All numbers and figures from the stored results | `experiments/paper_numbers_and_figures.sh` | all |

The last script needs no training: it reads the result files and checkpoints in `runs/` and writes `paper/data/*.json` and `paper/figs/fig1.pdf`–`fig6.pdf`. The versions of these outputs used for the paper are included.

**Seed means and validation-selected models.** The tables of arXiv v1 report means over seeds; `compute_numbers.py` and `compute_numbers2.py` produce them (`paper/data/numbers_G.json`, `paper/data/numbers_rest.json`).
`compute_selected.py` instead reports, for every method, the single model selected on validation data, never on test data: static operators by the held-out Group A loss (`heldout_A`), quantum models by the training loss over the first 10 steps, and dynamics models by the full-horizon rollout loss on the last 32 training trajectories (`val_sparse` for KAFlow, `val_full` for FNO). Its output, including the selected seeds, is `paper/data/selected.json`, and `paper/figs/` holds the figures drawn from it.

## Trained models and results

| Path | Contents |
|---|---|
| `runs/synth/G{1,2,3}_U2opd_s{0..5}` | KAFlow on G1–G3 (`.pt` checkpoint, `.json` losses, support and learned terms) |
| `runs/synth/G{1,2,3}_U2op_s{0..5}` | the same runs without the dense candidate; the symbolic distillation starts from these checkpoints. The dense candidate is never selected on G1–G3, so both give identical losses |
| `runs/synth/G{1,2,3}_U2sym_s*`, `runs/synth/G3_U2symnl_s*` | distilled closed forms (linear part; cubic nonlinearity of G3) |
| `runs/synth/G{1,2,3}_U_s*` | ablation without the position-dependent low-rank term |
| `runs/synth/H1_R{2,4,8,16}d_s{0,1,2}` | KAFlow on the variable-order fractional operator with $R$ separable terms |
| `runs/synthH/H1_{kano,fno}_s*.json` | KANO, KANO with an MLP symbol, and FNO trained with the official recipes |
| `runs/quantum/{dw,nlse}_{full,posmom,pos}_U3_s{0,1,2}` | KAFlow-Q; `*_U3sym_*.json` hold the distilled NLSE models |
| `runs/{eq}_U2_s{0..5}` | KAFlow on the evolution equations |
| `runs/{eq}_fno_tuned_s{0,1,2}`, `runs/{eq}_fno_sweep.jsonl` | tuned FNO baselines and their sweeps |
| `runs/sindy/{eq}_n0.json` | PDE-FIND and Weak SINDy |
| `runs/generality_check.json`, `runs/speed.json` | separation ranks and timings |

Checkpoints are PyTorch state dicts; the scripts in `paper/scripts/` show how each model is rebuilt from its training data statistics and loaded.

## Repository layout

| Path | Contents |
|---|---|
| `kafu.py` | dictionary symbols (`DictSymbol`), position functions (`XFn`), gauge-fixed nonlinearity (`FullProjPhi`), static operator (`KAFlowUOp2`), time-continuous model (`KAFlowU`) |
| `kaf.py` | B-spline utilities and the separable Kolmogorov–Arnold nonlinearity (`SepPhi`) |
| `bench/kaf_q.py` | quantum model (Strang split step, nonnegative kinetic dictionary, Kerr term) |
| `pde1d.py` | data generator for the evolution equations (pseudo-spectral Lawson RK4, 2/3 dealiasing) |
| `train_u2.py` | training on evolution equations (rollout curriculum, sparse support selection) |
| `bench/run_u_static2.py`, `bench/run_u_static.py` | training on static operators |
| `bench/symbolic_op.py`, `bench/symbolic_op_nl.py`, `bench/quantum_symbolic.py` | automatic symbolic distillation |
| `bench/run_quantum3.py` | quantum benchmarks |
| `bench/synthH_data.py` | variable-order fractional operator data |
| `fno_tune.py`, `baselines.py`, `bench/sindy_baseline.py` | FNO and SINDy baselines |
| `bench/kano_official_train.py`, `bench/fno_official_train.py` | KANO and FNO with the official recipes |
| `bench/kano_verify.py`, `bench/verify_baselines.py`, `bench/interp_test.py` | reproduction of the published KANO results |
| `bench/generality_check.py`, `bench/speed.py` | separation rank and cost |
| `paper/scripts/` | numbers and figures of the paper |
| `tools/extract_kano.py` | regenerates the KANO model classes from a local KANO clone |

Scripts locate the repository from their own path; set `KAFLOW_ROOT` to use another location. `DEVICE` (`cpu` or `cuda`) selects the device. Results can differ in the last digits across hardware and thread counts.

## License

MIT (see `LICENSE`). The KANO code and data are not part of this repository and remain under the terms of their authors.

## Citation

```bibtex
@article{cheon2026kaflow,
  title   = {Low-Rank {Kolmogorov--Arnold} Generator Flows for Interpretable Operator Learning},
  author  = {Cheon, Minjong},
  journal = {arXiv preprint arXiv:2610.13028},
  year    = {2026}
}
```

Contact: Minjong Cheon, Department of Computer Science, Korea National Open University (jmj2316@mail.knou.ac.kr)
