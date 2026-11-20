"""Run one multi-fidelity BO policy on one benchmark problem (benchmarks.py).

Everything runs in this process: a file-backed LocalHeroClient queue and a
LocalFunctionManager that evaluates the queued points with the Python
functions, so no scheduler or Hero account is needed.

Policies (all on the same SMT co-kriging model, LF cost 1, HF cost = ratio):

  sfbo       single-fidelity BO on HF only, from the same initial HF points
  mfsego     MF-SEGO (hero_mfsego.py): HF EI for location, fidelity by
             variance reduction per cost**power, promotion of LF-only points
  promotion  promotion-first (hero_mf_promotion.py): screen new points at LF,
             promote the best LF point to HF by EI per cost, streak guard
  mfkg       multi-fidelity knowledge gradient    (hero_mf_information.py)
  mfmes      multi-fidelity max-value entropy search (hero_mf_information.py)

Cost is reported in HF-evaluation equivalents spent after the initial design,
and regret is the best HF value evaluated so far minus the known minimum.

Examples
  python run_benchmark.py --problem rosenbrock --policy mfmes
  python run_benchmark.py --problem rastrigin --policy promotion --seed 40 --out r.json
  python run_benchmark.py --problem rosenbrock --ndim 10 --n-init-hf 11 --policy mfsego
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_computing.datasets import ContinuousVariable  # noqa: E402
from adaptive_computing.drivers import (ActiveLoopDriverHeroMFInfo,  # noqa: E402
                                        ActiveLoopDriverHeroMFPromotion,
                                        ActiveLoopDriverHeroMFSEGO)
from adaptive_computing.local_hero import LocalHeroClient  # noqa: E402
from benchmarks import PROBLEMS  # noqa: E402
from local_manager import LocalFunctionManager  # noqa: E402

POLICIES = ("sfbo", "mfsego", "promotion", "mfkg", "mfmes")


class _AlwaysHF(ActiveLoopDriverHeroMFSEGO):
    """HF EI on the highest level only; with one fidelity this is plain BO."""

    def get_next_sample(self):
        kb_s, kb_d = self._build_kriging_believer_surrogate()
        x = np.atleast_2d(self.sampler.minimize_acq_func(kb_s, kb_d, i_fidelity=self.n_fidelity - 1))
        self.last_candidate_source = "new"
        self.last_mfsego_scores = np.full(self.n_fidelity, np.nan)
        self.last_mfsego_eligible = np.ones(self.n_fidelity, dtype=bool)
        return x, self.n_fidelity - 1


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def run_benchmark(problem="rosenbrock", policy="mfmes", lf=None, ndim=None, hf_cost=None,
                  n_init_hf=8, budget=40.0, seed=38, batch=2, sego_power=2.0, streak_guard=10,
                  driver_kwargs=None, out=None, quiet=False):
    """Run one policy on one problem and return a summary dict.

    budget is in HF-evaluation equivalents after the initial design.  The
    initial design is n_init_hf random HF points; LF gets the same points plus
    two (np.random with this seed, so every policy starts from the same data).
    driver_kwargs are passed on to the driver (e.g. n_sobol for mfkg/mfmes).
    Returns dict(cost, regret, fidelities, n_hf, n_lf, final_regret, driver,
    elapsed, args); cost/regret are the best-so-far history.
    """
    assert problem in PROBLEMS, f"problem must be one of {sorted(PROBLEMS)}"
    assert policy in POLICIES, f"policy must be one of {POLICIES}"
    P = PROBLEMS[problem]
    lf = lf or P["default_lf"]
    assert lf in P["lfs"], f"{problem} LFs: {sorted(P['lfs'])}"
    ndim = ndim or P["ndim"]
    assert not (P["fixed_ndim"] and ndim != P["ndim"]), f"{problem} is {P['ndim']}D"
    func_lf, ratio = P["lfs"][lf]
    hf_cost = float(hf_cost or ratio)
    assert n_init_hf >= ndim + 1, f"AC fits the GP only with >= ndim + 1 = {ndim + 1} points per fidelity"
    args = dict(problem=problem, policy=policy, lf=lf, ndim=ndim, hf_cost=hf_cost, n_init_hf=n_init_hf,
                budget=budget, seed=seed, batch=batch, sego_power=sego_power, streak_guard=streak_guard)

    sf = policy == "sfbo"
    tmp = None
    if out:
        db = out + ".db.json"
    else:
        tmp = tempfile.TemporaryDirectory()
        db = os.path.join(tmp.name, "hero_db.json")
    for f in (db, db + ".lock"):
        if os.path.exists(f):
            os.remove(f)

    sink = io.StringIO() if quiet else None
    with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
        hero = LocalHeroClient(db_path=db)
        manager = LocalFunctionManager(hero_client=hero,
                                       simulation_functions=[P["hf"]] if sf else [func_lf, P["hf"]],
                                       machine_name="local", max_workers=4)
        common = dict(simulations=[None] if sf else [None, None],
                      fidelity_costs=[hf_cost] if sf else [1.0, hf_cost],
                      params=[ContinuousVariable(min=0, max=1)] * ndim,
                      machine_names=[manager.machine_name], output_field_path="y_data",
                      surrogate="SMT_GP", blocking=False, inline_manager=manager,
                      hero_client=hero, reference_minimum=P["y_min"], **(driver_kwargs or {}))
        if policy in ("mfkg", "mfmes"):
            drv = ActiveLoopDriverHeroMFInfo(acquisition=policy[2:], seed=seed, **common)
        elif policy == "mfsego":
            drv = ActiveLoopDriverHeroMFSEGO(mfsego_cost_power=sego_power, **common)
        elif policy == "promotion":
            drv = ActiveLoopDriverHeroMFPromotion(max_low_fidelity_streak=streak_guard, **common)
        else:
            drv = _AlwaysHF(mfsego_cost_power=1.0, promotion_ei_fraction=None, **common)

        drv.sampler._rand_seed = seed
        drv.init_sampler._rand_seed = seed
        np.random.seed(seed)
        x_hf = np.atleast_2d(np.random.rand(n_init_hf, ndim))
        x_lf = np.vstack([x_hf, np.random.rand(2, ndim)])
        if sf:
            drv.add_samples(x_hf, i_fidelity=0)
            manager.run_until_done(i_fidelities=[0])
        else:
            drv.add_samples(x_lf, i_fidelity=0)
            drv.add_samples(x_hf, i_fidelity=1)
            manager.run_until_done(i_fidelities=[0, 1])
        drv.hero_wait_for_data_and_train()

        t0 = time.time()
        while drv.cumulative_adaptive_cost < budget * hf_cost:
            drv.run(N_steps=batch, batch_size=batch)
            if not quiet:
                print(f"[progress] {problem} {policy} seed={seed} "
                      f"HF-eq={drv.cumulative_adaptive_cost / hf_cost:.1f} "
                      f"regret={drv.cost_regret_pairs[-1][1]:.4g} elapsed={time.time() - t0:.0f}s", flush=True)

    cr = np.array(drv.total_cost_regret_pairs, dtype=float)
    cost = (cr[:, 0] - cr[0, 0]) / hf_cost
    fids = [1] * len(drv.selection_history) if sf else \
        [1 if 1 in s["submitted_fidelities"] else 0 for s in drv.selection_history]
    res = dict(args=args, cost=cost, regret=cr[:, 1], fidelities=fids, n_hf=int(sum(fids)),
               n_lf=int(len(fids) - sum(fids)), final_regret=float(cr[-1, 1]),
               elapsed=time.time() - t0, driver=drv)
    if out:
        with open(out, "w") as f:
            json.dump(_clean({k: v for k, v in res.items() if k != "driver"}), f)
    if tmp is not None:
        tmp.cleanup()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", default="rosenbrock", choices=sorted(PROBLEMS))
    ap.add_argument("--policy", default="mfmes", choices=POLICIES)
    ap.add_argument("--lf", default=None, help="LF variant (default: hartmann phk, rosenbrock f3, rastrigin f2)")
    ap.add_argument("--ndim", type=int, default=None, help="dimension for rosenbrock / rastrigin (default 5)")
    ap.add_argument("--hf-cost", type=float, default=None, help="HF:LF cost ratio (default: the LF's)")
    ap.add_argument("--n-init-hf", type=int, default=8)
    ap.add_argument("--budget", type=float, default=40.0, help="HF-evaluation equivalents after the initial design")
    ap.add_argument("--seed", type=int, default=38)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--sego-power", type=float, default=2.0)
    ap.add_argument("--streak-guard", type=int, default=10)
    ap.add_argument("--out", default=None, help="write the history as JSON here")
    a = ap.parse_args()
    r = run_benchmark(problem=a.problem, policy=a.policy, lf=a.lf, ndim=a.ndim, hf_cost=a.hf_cost,
                      n_init_hf=a.n_init_hf, budget=a.budget, seed=a.seed, batch=a.batch,
                      sego_power=a.sego_power, streak_guard=a.streak_guard, out=a.out)
    print(f"DONE {a.problem} {a.policy} seed={a.seed}: regret {r['final_regret']:.4g} after "
          f"{r['cost'][-1]:.1f} HF-eq, {r['n_hf']} HF / {r['n_lf']} LF selections, {r['elapsed']:.0f}s")


if __name__ == "__main__":
    main()
