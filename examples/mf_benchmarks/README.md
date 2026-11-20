# mf_benchmarks: multi-fidelity BO policies on standard benchmark problems

Runs one fidelity-selection policy on one two-fidelity test problem, entirely in
one process (a file-backed `LocalHeroClient` queue plus a `LocalFunctionManager`
that evaluates queued points with Python functions). No scheduler or Hero
account is needed.

```bash
cd examples/mf_benchmarks
python run_benchmark.py --problem rosenbrock --policy mfmes          # 5D, LF f3, HF:LF cost 10:1
python run_benchmark.py --problem rastrigin  --policy promotion --seed 40
python run_benchmark.py --problem rosenbrock --ndim 10 --n-init-hf 11 --policy mfsego
python run_benchmark.py --help
```

or from Python:

```python
from run_benchmark import run_benchmark
r = run_benchmark(problem="rosenbrock", policy="mfmes", budget=40, seed=38)
r["cost"], r["regret"], r["n_hf"], r["n_lf"]
```

Cost is counted in HF-evaluation equivalents spent after the initial design
(LF cost 1, HF cost = the cost ratio). Regret is the best HF value evaluated so
far minus the known minimum. A full 40-HF-equivalent run takes minutes for
single-fidelity BO and 0.5 to 3 hours for the multi-fidelity policies in 5D
(longer in 10D), so run seed sweeps on a cluster, one process per seed.

## Files

| File | Contents |
|---|---|
| `benchmarks.py` | Test problems and the `PROBLEMS` registry |
| `run_benchmark.py` | `run_benchmark()` and its command line |
| `local_manager.py` | `LocalFunctionManager`: evaluates queued tasks in-process |

## Problems (`benchmarks.py`)

| Problem | D | LFs (HF:LF cost) | Notes |
|---|---|---|---|
| `hartmann` | 6 | `phk` (default), `close`, `forrester` (10:1, none published) | `phk` is the Park, Haftka & Kim (2016) LF. It tracks HF within each deep basin but with different scale factors, so constant-rho co-kriging ranks the basins backwards. `close` and `forrester` have their minimum at the HF minimizer. |
| `rosenbrock` | any (5) | `f2` (2:1), `f3` (10:1, default) | AVT-331 L1 benchmark MF2 |
| `rastrigin` | any (5) | `f2` (16:1, default), `f3` (256:1) | AVT-331 L1 benchmark MF3, shifted and rotated |

The AVT-331 problems are from Mainini et al. (2022), arXiv:2204.07867,
transcribed from the authors' reference code
(gitlab.com/qudo046/avt-331-l1-benchmarks) with the paper's costs.

## Policies (`--policy`)

All share the same SMT co-kriging (MFK) surrogate.

| Policy | Driver | Selection rule |
|---|---|---|
| `sfbo` | `_AlwaysHF` (here) | Single-fidelity BO on HF only, from the same initial HF points |
| `mfsego` | `ActiveLoopDriverHeroMFSEGO` | HF EI for location; fidelity by variance reduction per cost^power; promotion of LF-only points |
| `promotion` | `ActiveLoopDriverHeroMFPromotion` | Screen new points at LF; promote the best LF point to HF by EI per cost; streak guard (10) |
| `mfkg` | `ActiveLoopDriverHeroMFInfo(acquisition="kg")` | Multi-fidelity knowledge gradient per cost |
| `mfmes` | `ActiveLoopDriverHeroMFInfo(acquisition="mes")` | Multi-fidelity max-value entropy search per cost; no tuned parameters |

## Results so far

15 paired seeds (38 to 52) per configuration, budget 40 HF-equivalents, 8
initial HF points (11 in 10D). Median best HF regret at 40, and wins-losses
against single-fidelity BO on the same seeds.

| Configuration | Single-fidelity | MF-SEGO | Promotion | MF-KG | MF-MES |
|---|---|---|---|---|---|
| Rosenbrock 5D, f3, 10:1 | 26.6 | 2.8 (15-0) | 3.7 (15-0) | 2.1 (14-1) | **0.25** (15-0) |
| Rastrigin 5D, f2, 16:1 | 0.20 | 0.083 (14-1) | 0.068 (13-2) | 0.081 (13-2) | **0.041** (15-0) |
| Rosenbrock 10D, f3, 10:1 | 238 | 59 (13-2) | 57 (15-0) | 138 (8-5)* | **30** (10-0)* |
| Rosenbrock 5D, f2, 2:1 | 24.0 | **8.1** (14-1) | 8.8 (12-3) | 258 (0-15) | 147 (5-10) |

\* On the 13 and 10 seeds that reached 40 HF-equivalents.

On the Hartmann `phk` pair (cost 3, 10 and 30, 8 or 12 initial points) no
policy beats single-fidelity BO at 40 HF-equivalents. MF-MES is far ahead
between 5 and 30 HF-equivalents at cost 10 and 30 (15 of 15 seeds at 10 and
20).

In short:

- **HF:LF cost of 10 or more:** MF-MES is the best or tied-best policy.
- **Low cost ratio (2-3:1):** MF-KG and MF-MES almost never pay for HF
  (value per cost favours a correlated LF), so their best evaluated HF point
  barely improves. MF-SEGO and promotion-first, which force HF evaluations,
  still win.
- **MF-KG** never beat MF-MES.
- **When the LF misleads** (Hartmann `phk`), no policy beats single-fidelity BO
  at the full budget.
