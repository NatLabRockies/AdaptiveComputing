#!/usr/bin/env python3
"""
controller.py — Rental Car Electrification Co-Scientist (HPC_onsite variant)
=============================================================================
Identical to rental_agent.py in co-scientist logic but uses LocalHeroClient
+ LocalHPCManager.run_until_done() inline instead of the AC MCP HTTP server.
No HTTP server to start or stop; the inline manager submits and polls SLURM
jobs directly from this process.  LangGraph SQLite checkpointer handles crash
recovery.

Graph
-----
    clarify ─────────────────────────────────────────────────────────────┐
        │                                                                 │
        ▼                                             (follow-up)        │
       plan ◀─── (plan_feedback from either approve node)                │
        │                                                                 │
        ▼                                                                 │
  approve_direction ──► search_registry ──► approve_concrete             │
                                                  │                      │
                                            execute_step (loop)          │
                                                  │                      │
                                       synthesize_and_explain            │
                                                  │                      │
                                            ask_followup ────────────────┘
                                                  │ (empty input)
                                                 END

Usage
-----
    python controller.py
    python controller.py "What storage minimizes cost for 5000 EVs/day?"
    python controller.py "Compare Moderate vs Aggressive utility rates."
"""

import argparse
import os
import sys
import uuid as _uuid_module
from pathlib import Path
from typing import List, Literal, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_AGENT_DIR = Path(__file__).parent.resolve()
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

# Set AC_MCP_DIR so ac_mcp/registry.py stores experiments in this directory.
os.environ.setdefault("AC_MCP_DIR", str(_AGENT_DIR))

try:
    from adaptive_computing.utils import load_env_file
    load_env_file(str(_AGENT_DIR / "env_vars.txt"))
except (FileNotFoundError, Exception):
    pass

# ---------------------------------------------------------------------------
# HPC_onsite-specific imports
# ---------------------------------------------------------------------------
import numpy as np

from adaptive_computing.local_hero import LocalHeroClient
from adaptive_computing.datasets import OrderedVariable
from adaptive_computing.drivers import ActiveLoopDriverHero
from ac_mcp import registry
from ac_mcp.param_builder import (
    build_ac_params,
    build_task_formatter,
    build_evaluation_formatter,
)
from ac_mcp.run_manager import _extract_results
from manager import create_manager

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
_OUTPUT_LABEL  = "Daily cost (USD)"
_UTILITY_RATES = ["Moderate", "Aggressive"]

# Default exploration space — used by _run_exploration_step when param_specs is empty.
_DEFAULT_EXPLORATION_SPECS = [
    {"name": "utility_rate",        "type": "categorical", "categories": _UTILITY_RATES},
    {"name": "storage",             "type": "continuous",  "min": 0.0,   "max": 100.0},
    {"name": "number_of_daily_evs", "type": "continuous",  "min": 10.0,  "max": 10000.0},
    {"name": "return_soc",          "type": "continuous",  "min": 25.0,  "max": 55.0},
]

# ---------------------------------------------------------------------------
# Session globals  (reset at the start of each run_agent() call)
# ---------------------------------------------------------------------------
MACHINE_NAME = "local"

_SESSION_HERO: "LocalHeroClient | None" = None
_SESSION_MANAGER = None

# ---------------------------------------------------------------------------
# Checkpoint context (populated when running under co_scientist.py)
# ---------------------------------------------------------------------------
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

_CHAT_ID: Optional[str] = None
_CHECKPOINT_FILE: Optional[str] = None
_CHECKPOINT_STATE: dict = {}


def _write_checkpoint(**updates) -> None:
    """Update and persist the checkpoint if running under co_scientist."""
    if not _CHAT_ID or not _CHECKPOINT_FILE:
        return
    _CHECKPOINT_STATE.update(updates)
    try:
        import chat_registry
        chat_registry.write_checkpoint(_CHECKPOINT_FILE, dict(_CHECKPOINT_STATE))
    except Exception as exc:
        print("[checkpoint] Warning: {}".format(exc))


# ---------------------------------------------------------------------------
# 1. Schemas  (identical to rental_agent.py)
# ---------------------------------------------------------------------------

class ParamSpec(BaseModel):
    """One parameter in a Bayesian optimisation search space."""
    name:       str                                             = Field(..., description="Parameter name")
    type:       Literal["continuous", "ordered", "categorical"] = Field(..., description="Variable type")
    min:        Optional[float]      = Field(None, description="Lower bound (continuous / ordered)")
    max:        Optional[float]      = Field(None, description="Upper bound (continuous / ordered)")
    categories: Optional[List[str]] = Field(None, description="Allowed values (categorical)")


class PlanStep(BaseModel):
    tool: Literal["run_simulation", "run_exploration", "run_optimization", "evaluate_surrogate"] = Field(
        description=(
            "run_simulation: evaluate one fixed configuration; "
            "run_exploration: LHS sampling over parameter space to survey costs; "
            "run_optimization: Bayesian optimization to find the minimum-cost configuration; "
            "evaluate_surrogate: query the trained surrogate from a prior run_optimization "
            "to predict cost at a specific point — no simulation or HPC needed."
        )
    )
    purpose: str = Field(description="One-sentence explanation of why this step is in the plan.")
    label: Optional[str] = Field(None, description="Short human-readable label for this step.")
    reasoning: Optional[str] = Field(None, description="Deeper scientific rationale (optional).")

    # All parameter values for this step:
    #   run_simulation / evaluate_surrogate: every parameter goes here.
    #   run_optimization: parameters held FIXED go here; free ones go in param_specs.
    #   run_exploration: empty {} or a subset to fix some params during the sweep.
    fixed_context: Optional[dict] = Field(
        None,
        description=(
            "Parameter values held constant for this step. "
            "run_simulation/evaluate_surrogate: all 4 params. "
            "run_optimization: only the parameters NOT being optimised. "
            "run_exploration: empty or partially constrained."
        ),
    )

    # Free variables to optimise (run_optimization) or sweep (run_exploration).
    param_specs: Optional[List[ParamSpec]] = Field(
        None,
        description=(
            "Parameters to optimise (run_optimization) or explore (run_exploration). "
            "Omit for run_simulation / evaluate_surrogate."
        ),
    )

    # Exploration
    n_exploration_samples: Optional[int] = Field(
        None, description="LHS samples for run_exploration (default 20)."
    )

    # BO knobs (run_optimization only)
    n_init_samples: Optional[int] = Field(
        None, description="LHS warm-up samples before BO (default 3; auto-skipped on warm-start)."
    )
    n_bo_batches: Optional[int] = Field(
        None, description="Serial BO rounds (default 1). Total evals = n_bo_batches × n_parallel_per_batch."
    )
    n_parallel_per_batch: Optional[int] = Field(
        None,
        description=(
            "Evaluations per BO round (default 1=sequential). "
            ">1 = parallel batch (faster wall-clock, less sample-efficient)."
        ),
    )


class ResearchPlan(BaseModel):
    steps:       List[PlanStep] = Field(description="Ordered list of steps to execute.")
    description: str            = Field(description="Single-line summary of the full plan.")
    reasoning:   str            = Field(description="Scientific rationale for the plan.")


class ClarificationDecision(BaseModel):
    needs_clarification: bool      = Field(description="True if clarifying questions are needed.")
    questions:           List[str] = Field(default_factory=list)


class ReuseStepPatch(BaseModel):
    step_index:           int           = Field(..., description="1-based step index from the plan")
    use_prior_data:       bool          = Field(True,  description="True=warm-start from prior data; False=discard and run fresh with LHS")
    n_bo_batches:         Optional[int] = Field(None,  description="Override serial BO rounds; None=keep original planned value")
    n_parallel_per_batch: Optional[int] = Field(None,  description="Override parallel evaluations per round; None=keep original planned value")


class ReusePatchResult(BaseModel):
    patches: List[ReuseStepPatch] = Field(
        ..., description="One patch per warm-start step (include ALL steps, even unmentioned ones)"
    )


class AgentState(TypedDict):
    user_request:          str
    conversation_history:  list
    clarification_context: Optional[str]
    plan_feedback:         Optional[str]
    plan_steps:            list
    plan_reasoning:        Optional[str]
    plan_description:      Optional[str]
    completed_steps:       list
    accumulated_results:   list
    reuse_notes:           list
    status:                str
    error:                 Optional[str]
    response:              Optional[str]


# ---------------------------------------------------------------------------
# 2. System prompts  (identical to rental_agent.py)
# ---------------------------------------------------------------------------

_SIMULATOR_CONTEXT = """\
## Simulator Context
You are advising on a pre-built, fixed black-box rental car electrification model.
The following properties are FIXED and cannot be changed:

  Facility     : Airport rental car center with EV fleet
  Objective    : Minimize total daily energy cost (USD)
  Metric       : cost — total daily cost to charge the fleet

The ONLY parameters the agent can control, with their types and ranges:

  utility_rate          categorical   "Moderate" | "Aggressive"
                                      Charging tariff structure.
  storage              continuous    0 – 100  (percentage of max battery storage)
                                      0 = grid-only (no storage), 100 = maximum storage.
  number_of_daily_evs   continuous    10 – 10000
                                      Average daily fleet throughput.
  return_soc            continuous    25 – 55
                                      Average state-of-charge when vehicles return.

For run_simulation and evaluate_surrogate: specify all parameters in fixed_context.
For run_optimization: free parameters go in param_specs; fixed ones go in fixed_context.
For run_exploration: specify parameters to sweep in param_specs (empty = sweep all 4).

Variable types supported by the optimizer:
  continuous  — real-valued float; specify min and max
  ordered     — integer; specify min and max
  categorical — discrete string set; specify categories list

All questions and plans must be grounded in these four parameters and in
the scientific goal of understanding and minimizing cost.
"""

_CLARIFY_SYSTEM_PROMPT = _SIMULATOR_CONTEXT + """
## Your task
Decide whether 1–3 targeted clarifying questions would meaningfully improve
the research plan.  Ask only when the answer would change which tool gets
called or which parameter values get explored.

Good reasons to ask:
  - The user mentions multiple utility rates without saying whether to
    compare them or focus on one
  - The demand scale isn't specified and it matters for the question
  - Ambiguity about whether to fix some parameters or sweep them

Do NOT ask about:
  - Anything not listed in the Simulator Context above
  - Details already implied by the request
  - Prior conversation context that already resolves the ambiguity

If in doubt, set needs_clarification=False.
"""

_PLAN_SYSTEM_PROMPT = _SIMULATOR_CONTEXT + """
## Your task
Produce a creative multi-step research plan that fully addresses the user's
request using only the available tools and controllable parameters above.

## Available Tools

### run_simulation
Evaluate one fixed configuration. Set:
  fixed_context: dict with ALL four parameter values, e.g.
    {"utility_rate": "Moderate", "storage": 50.0,
     "number_of_daily_evs": 1000.0, "return_soc": 40.0}
  label, purpose.

### run_exploration
Draw N LHS samples to survey cost across the parameter space. Set:
  param_specs: parameters to sweep (omit for full 4-param sweep), e.g.
    [{"name":"storage","type":"continuous","min":0,"max":100}]
  fixed_context: any parameters to hold fixed during the sweep (or {})
  n_exploration_samples, label, purpose.

### run_optimization
Bayesian-optimize any subset of parameters jointly. Set:
  param_specs: FREE variables to optimise, each as a dict:
    {"name": "storage",             "type": "continuous",  "min": 0.0,   "max": 100.0}
    {"name": "number_of_daily_evs", "type": "continuous",  "min": 10.0,  "max": 10000.0}
    {"name": "return_soc",          "type": "continuous",  "min": 25.0,  "max": 55.0}
    {"name": "utility_rate",        "type": "categorical", "categories": ["Moderate","Aggressive"]}
  fixed_context: parameters held FIXED (must include all 4 minus those in param_specs)
  n_init_samples, n_bo_batches, n_parallel_per_batch, label, purpose.

  At least one parameter must be in param_specs.
  Total BO evaluations = n_bo_batches × n_parallel_per_batch.
  n_parallel_per_batch=1 (default) is sequential (most sample-efficient).
  n_parallel_per_batch>1 is a parallel batch (faster wall-clock, less efficient).
  Ask the user when they specify a batch structure; otherwise default to
  n_bo_batches=1, n_parallel_per_batch=1.

### evaluate_surrogate
Query the trained surrogate from a prior run_optimization — no simulation needed. Set:
  fixed_context: the exact parameter point to evaluate (all 4 params)
  label, purpose.
Pair with a prior run_optimization covering the same parameter space.

## Creative Composition Examples
  Compare two configs     → 2 × run_simulation (different fixed_contexts)
  Sweep utility rates     → 2 × run_simulation (Moderate vs Aggressive, rest fixed)
  Survey full space       → run_exploration (param_specs=[], or all 4 params)
  Optimize storage + SOC  → run_optimization(param_specs=[storage,return_soc],
                              fixed_context={utility_rate,number_of_daily_evs})
  Optimize everything     → run_optimization(param_specs=[all 4], fixed_context={})
  Baseline then optimize  → run_simulation(storage=0) + run_optimization
  Optimize then predict   → run_optimization + evaluate_surrogate
  3 batches × 4 parallel  → run_optimization(n_bo_batches=3, n_parallel_per_batch=4)

Set 'reasoning' with scientific rationale grounded in the parameter ranges and
what the simulations are expected to reveal.
Set 'description' to a one-line summary of the full plan.
"""

_EXPLAIN_SYSTEM_PROMPT = _SIMULATOR_CONTEXT + """
## Your task
Synthesize a detailed expert explanation of the results from the research
plan that was just executed.  Cover:

1. What was investigated and why (reference the plan reasoning).
2. What the observed cost numbers reveal about how each parameter affects cost.
3. If multiple configs were compared: which was cheapest and what does that
   suggest about the underlying cost drivers?
4. If optimization was performed: which parameter values were selected as
   optimal and what does that imply about the cost landscape?
5. If exploration was performed: describe the shape of the cost landscape —
   which parameters dominate, are there apparent interactions?
6. Concrete, specific recommendations for follow-up experiments.

Do NOT assume or state knowledge of the underlying formula or model internals.
Reason purely from the simulation results that were observed.
Anticipate likely follow-up questions and address them proactively.
Be specific — cite the actual numerical values from the results.
"""

_NEGOTIATE_REUSE_SYSTEM_PROMPT = """\
You are parsing user preferences for warm-start Bayesian optimization.

Context: Prior optimization data was found for one or more steps in the
research plan.  LHS initialization will be skipped automatically when prior
data exists — existing samples are seeded into the BO surrogate.  The user
was asked whether they want to reuse that data and how many additional BO
evaluations to run, specified as M serial batches × N parallel per batch.

Parse their response into one ReuseStepPatch per warm-start step.

Rules:
  - Plain number N (e.g. "3" or "3 samples")  → n_bo_batches=1, n_parallel_per_batch=N,
    use_prior_data=True for ALL steps.  Treat a plain number as 1 batch of N parallel.
  - "M batches of N" / "M×N" / "M rounds of N" → n_bo_batches=M, n_parallel_per_batch=N.
  - "N sequential" / "N steps one at a time"  → n_bo_batches=N, n_parallel_per_batch=1.
  - "fresh" / "from scratch"                  → use_prior_data=False for ALL steps.
  - Per-step (e.g. "step 2: 3 samples, step 3: fresh") → apply only to those steps;
    leave other steps with use_prior_data=True, n_bo_batches/n_parallel_per_batch=None.
  - "keep" / "yes" / "default"               → use_prior_data=True, both batch fields=None.
  - All counts must be >= 1 when set.
  - Always return one patch per warm-start step, even if the user didn't mention it
    (use defaults: use_prior_data=True, n_bo_batches=None, n_parallel_per_batch=None).
"""

_APPROVE_YES = {"yes", "y", "ok", "sure", "looks good", "approved",
                "proceed", "go", "go ahead", "sounds good", ""}


# ---------------------------------------------------------------------------
# 3. LLM factory  (identical to rental_agent.py)
# ---------------------------------------------------------------------------

def _get_llm():
    provider = os.environ.get("RENTAL_LLM_PROVIDER", "litellm").lower()
    if provider == "litellm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.environ.get("LITELLM_MODEL", "gpt-5-mini"),
            base_url=os.environ["LITELLM_ENDPOINT"],
            api_key=os.environ["LITELLM_API_KEY"],
        )
    elif provider == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-5-sonnet-20241022")
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )


# ---------------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------------

def _fetch_registry_summary() -> str:
    """Return a plain-text summary of the local experiment registry."""
    try:
        experiments = registry.list_entries()
        if not experiments:
            return "Registry: empty (no experiments run yet)."
        completed = [e for e in experiments if e.get("run_status") == "completed"]
        in_prog   = [e for e in experiments if e.get("run_status") != "completed"]
        lines = ["Registry ({} completed, {} in-progress):".format(
            len(completed), len(in_prog))]
        for e in completed[:30]:
            eid      = e.get("id", "?")[:8]
            name     = e.get("name", "?")
            etype    = e.get("experiment_type", "?")
            best_y   = e.get("best_y")
            best_str = " best=${:.0f}".format(-best_y) if best_y is not None else ""
            fixed    = e.get("fixed_context", {})
            lines.append("  [{}] \"{}\" [{}] fixed={}{}".format(
                eid, name, etype, fixed, best_str))
        if in_prog:
            lines.append("  ({} in-progress not shown)".format(len(in_prog)))
        return "\n".join(lines)
    except Exception as exc:
        return "Registry: unavailable ({})".format(exc)


def _lhs_jobs(n_samples: int, seed: int = 42) -> list:
    """Generate Latin Hypercube Samples as explicit parameter dicts."""
    import random
    rng = random.Random(seed)

    def _lhs1(n):
        perms = list(range(n))
        rng.shuffle(perms)
        return [(p + rng.random()) / n for p in perms]

    utility_u = _lhs1(n_samples)
    storage_u = _lhs1(n_samples)
    evs_u     = _lhs1(n_samples)
    soc_u     = _lhs1(n_samples)

    jobs = []
    for i in range(n_samples):
        ur_idx = min(int(utility_u[i] * len(_UTILITY_RATES)), len(_UTILITY_RATES) - 1)
        jobs.append({
            "utility_rate":        _UTILITY_RATES[ur_idx],
            "storage":            round(storage_u[i] * 100.0, 4),
            "number_of_daily_evs": round(10.0 + evs_u[i] * (10000.0 - 10.0), 4),
            "return_soc":          round(25.0 + soc_u[i] * (55.0 - 25.0), 4),
        })
    return jobs


def _fmt_specs(specs: list) -> str:
    """Format a list of param_specs for display."""
    parts = []
    for ps in specs:
        if isinstance(ps, dict):
            name, ptype = ps.get("name","?"), ps.get("type","")
        else:
            name, ptype = ps.name, ps.type
            ps = ps.model_dump()
        if ptype in ("continuous", "ordered"):
            parts.append("{}∈[{},{}]".format(name, ps.get("min","?"), ps.get("max","?")))
        else:
            parts.append("{}∈{}".format(name, ps.get("categories",[])))
    return "  ".join(parts)


def _decode_x(x_row, param_specs: list, fixed_context: dict) -> dict:
    """Decode a raw AC x_data row back to named parameter values."""
    result = dict(fixed_context)
    for j, spec in enumerate(param_specs):
        raw   = float(x_row[j])
        ptype = spec["type"].lower()
        if ptype == "categorical":
            result[spec["name"]] = spec["categories"][int(round(raw))]
        elif ptype == "ordered":
            result[spec["name"]] = int(round(raw))
        else:
            result[spec["name"]] = raw
    return result


# ---------------------------------------------------------------------------
# 4b. Inline step execution functions (replace MCP calls)
# ---------------------------------------------------------------------------

def _run_simulation_step(step: dict) -> dict:
    """Run one fixed-configuration evaluation via inline LocalHPCManager."""
    fixed_ctx = dict(step.get("fixed_context") or {})
    label     = step.get("label") or "sim-{}".format(
        "-".join(str(v) for v in list(fixed_ctx.values())[:3]))

    print("  Submitting: {}".format(
        "  ".join("{}={}".format(k, v) for k, v in sorted(fixed_ctx.items()))))

    entry = registry.find_matching_experiment(
        name="rental-car-eval", param_specs=[],
        fixed_context=fixed_ctx, experiment_type="evaluation",
    )
    if entry and entry.get("run_status") == "completed":
        print("  (reusing completed experiment {})".format(entry["id"][:8]))
        try:
            data  = registry.load_dataset(entry["id"])
            y_val = float(data["y_data"][0][0])
            cost  = -y_val
        except Exception:
            cost = None
        return {
            "data_points": [{"label": label, "params": fixed_ctx, "cost": cost}],
            "reuse_note": "Reused cached result for {}.".format(label),
            "error": None,
        }

    exp_id = registry.register_experiment(
        name=step.get("label") or "rental-car-eval",
        description=step.get("purpose", "single evaluation"),
        param_specs=[],
        fixed_context=fixed_ctx,
        output_label=_OUTPUT_LABEL,
        hpc_config_path="",
        experiment_type="evaluation",
    )

    formatter = build_evaluation_formatter([fixed_ctx], [_SESSION_MANAGER.machine_name])
    driver = ActiveLoopDriverHero(
        simulations=[None],
        params=[OrderedVariable(min_val=0, max_val=0)],
        machine_names=[_SESSION_MANAGER.machine_name],
        output_field_path="y_data",
        surrogate=None,
        blocking=False,
        task_formatter=formatter,
        hero_client=_SESSION_HERO,
    )

    try:
        driver.dataset.add_samples(np.array([[0.0]]), 0)
        _SESSION_MANAGER.run_until_done(i_fidelity=0)
        driver.dataset.hero_wait_for_data()

        y_val = float(driver.dataset.y_data[0][0][0])
        if np.isnan(y_val):
            raise ValueError("simulation returned NaN")
        cost = -y_val

        registry.save_dataset(exp_id, driver.dataset.x_data[0], driver.dataset.y_data[0])
        return {
            "data_points": [{"label": label, "params": fixed_ctx, "cost": cost}],
            "reuse_note": None, "error": None,
        }
    except Exception as exc:
        registry.update_entry(exp_id, run_status="error")
        return {"data_points": [], "reuse_note": None,
                "error": "Simulation error: {}".format(exc)}


def _run_exploration_step(step: dict) -> dict:
    """Submit N LHS jobs at once, wait, and collect results."""
    n_samples   = int(step.get("n_exploration_samples") or 20)
    label       = step.get("label") or "exploration-{}".format(n_samples)
    raw_specs   = step.get("param_specs") or []
    param_specs = ([ps.model_dump() if hasattr(ps, "model_dump") else dict(ps)
                    for ps in raw_specs]
                   if raw_specs else list(_DEFAULT_EXPLORATION_SPECS))
    fixed_ctx   = dict(step.get("fixed_context") or {})

    print("  Exploring: {} LHS samples  params=[{}]".format(
        n_samples, _fmt_specs(param_specs)))

    entry = registry.find_matching_experiment(
        name=step.get("label") or "rental-car-exploration",
        param_specs=param_specs,
        fixed_context=fixed_ctx, experiment_type="exploration",
    )
    if entry and entry.get("run_status") == "completed":
        print("  (reusing completed exploration {})".format(entry["id"][:8]))
        try:
            data = registry.load_dataset(entry["id"])
            data_points = []
            for i, (x_row, y_row) in enumerate(zip(data["x_data"], data["y_data"])):
                x_dict = _decode_x(x_row, param_specs, fixed_ctx)
                cost = -float(y_row[0]) if not np.isnan(float(y_row[0])) else None
                data_points.append({"label": "lhs-{}".format(i + 1),
                                     "params": x_dict, "cost": cost})
            return {"data_points": data_points,
                    "reuse_note": "Reused cached exploration {}.".format(label),
                    "error": None}
        except Exception:
            pass  # fall through to fresh run

    exp_id = registry.register_experiment(
        name=step.get("label") or "rental-car-exploration",
        description=step.get("purpose", "LHS exploration"),
        param_specs=param_specs,
        fixed_context=fixed_ctx,
        output_label=_OUTPUT_LABEL,
        hpc_config_path="",
        experiment_type="exploration",
    )

    jobs      = _lhs_jobs(n_samples)
    formatter = build_evaluation_formatter(jobs, [_SESSION_MANAGER.machine_name])
    driver    = ActiveLoopDriverHero(
        simulations=[None],
        params=[OrderedVariable(min_val=0, max_val=max(n_samples - 1, 1))],
        machine_names=[_SESSION_MANAGER.machine_name],
        output_field_path="y_data",
        surrogate=None,
        blocking=False,
        task_formatter=formatter,
        hero_client=_SESSION_HERO,
    )

    try:
        x_all = np.array([[float(i)] for i in range(n_samples)])
        driver.dataset.add_samples(x_all, 0)
        _SESSION_MANAGER.run_until_done(i_fidelity=0)
        driver.dataset.hero_wait_for_data()

        results, _, _ = _extract_results(driver, param_specs, fixed_ctx)
        data_points = []
        for i, r in enumerate(results):
            cost = -r["y"] if r.get("y") is not None else None
            data_points.append({"label": "lhs-{}".format(i + 1),
                                 "params": {**fixed_ctx, **r.get("x", {})},
                                 "cost": cost})

        registry.save_dataset(exp_id, driver.dataset.x_data[0], driver.dataset.y_data[0])
        return {"data_points": data_points, "reuse_note": None, "error": None}
    except Exception as exc:
        registry.update_entry(exp_id, run_status="error")
        return {"data_points": [], "reuse_note": None,
                "error": "Exploration error: {}".format(exc)}


def _run_optimization_step(step: dict) -> dict:
    """Run Bayesian optimization inline with run_until_done() per step."""
    raw_specs     = step.get("param_specs") or []
    param_specs   = [ps.model_dump() if hasattr(ps, "model_dump") else dict(ps)
                     for ps in raw_specs]
    fixed_context = dict(step.get("fixed_context") or {})
    opt_var_names = [s["name"] for s in param_specs]

    if not param_specs:
        return {"data_points": [], "reuse_note": None,
                "error": "run_optimization: param_specs is empty — nothing to optimize."}

    n_init_raw = step.get("n_init_samples")
    n_init     = 3 if n_init_raw is None else int(n_init_raw)
    n_batches  = int(step.get("n_bo_batches") or 1)
    n_parallel = int(step.get("n_parallel_per_batch") or 1)
    n_steps    = n_batches * n_parallel
    label      = step.get("label") or "opt-{}".format("+".join(opt_var_names))

    fixed_str = "  ".join("{}={}".format(k, v) for k, v in sorted(fixed_context.items()))
    print("  Optimizing [{}]: fixed=({})  (init={}, {}×{} BO = {} evals)".format(
        _fmt_specs(param_specs), fixed_str, n_init, n_batches, n_parallel, n_steps))

    fresh_run = bool(step.get("_fresh_run"))

    exp_id = registry.register_experiment(
        name=label,
        description=step.get("purpose", "Bayesian optimization"),
        param_specs=param_specs,
        fixed_context=fixed_context,
        output_label=_OUTPUT_LABEL,
        hpc_config_path="",
        experiment_type="optimization",
    )

    prior     = registry.find_reusable_data(
        param_specs=param_specs,
        fixed_context=fixed_context,
        experiment_type="optimization",
        exclude_id=exp_id,
    )
    use_prior = prior["n_valid"] > 0 and not fresh_run

    ac_params = build_ac_params(param_specs)
    formatter = build_task_formatter(param_specs, fixed_context,
                                     [_SESSION_MANAGER.machine_name])

    # inline_manager is not set here; we drive the BO loop manually so we can
    # submit n_parallel jobs per batch before calling run_until_done once.
    driver = ActiveLoopDriverHero(
        simulations=[None],
        params=ac_params,
        machine_names=[_SESSION_MANAGER.machine_name],
        output_field_path="y_data",
        surrogate="SMT_GP",
        acq_func="expected_improvement",
        blocking=False,
        task_formatter=formatter,
        hero_client=_SESSION_HERO,
    )

    try:
        n_warmup = 0

        if use_prior:
            n_dupes = prior.get("n_duplicates_removed", 0)
            src     = ", ".join(prior["source_ids"])
            print("  Auto warm-start: {} pts from [{}]{}".format(
                prior["n_valid"], src,
                " ({} dupes removed)".format(n_dupes) if n_dupes else ""))
            driver.dataset.add_known_samples(prior["x_valid"], prior["y_valid"], 0)
            driver.surrogate.train(driver.dataset)
            driver._bopt_initialized = True
            n_warmup = prior["n_valid"]
            registry.save_dataset(exp_id, driver.dataset.x_data[0],
                                   driver.dataset.y_data[0], set_completed=False)
        else:
            # LHS init: submit all points at once, single run_until_done call.
            print("  Warm-up: {} LHS jobs...".format(n_init))
            lhs_x = driver.init_sampler.get_sample(N_samples=n_init)
            driver.dataset.add_samples(lhs_x, 0)
            _write_checkpoint(status="waiting", n_pending=n_init)
            _SESSION_MANAGER.run_until_done(i_fidelity=0)
            driver.dataset.hero_wait_for_data()
            driver.surrogate.train(driver.dataset)
            driver._bopt_initialized = True
            n_warmup = n_init
            _write_checkpoint(status="active", n_pending=0)
            registry.save_dataset(exp_id, driver.dataset.x_data[0],
                                   driver.dataset.y_data[0], set_completed=False)

        # BO phase: submit n_parallel jobs per batch using the Kriging Believer
        # strategy, then run_until_done once per batch.  BayesianSampler.get_sample()
        # automatically builds KB phantoms for any pending (masked) points, so
        # successive calls within a batch explore diverse regions.  SLURM may
        # run the n_parallel jobs concurrently if resources are available.
        print("  BO: {} batch(es) × {} job(s) = {} evals total...".format(
            n_batches, n_parallel, n_steps))
        for batch in range(n_batches):
            for j in range(n_parallel):
                x, fi = driver.get_next_sample()
                driver.dataset.add_samples(x, fi)
            _write_checkpoint(status="waiting", n_pending=n_parallel)
            _SESSION_MANAGER.run_until_done(i_fidelity=0)
            driver.dataset.hero_wait_for_data()
            driver.surrogate.train(driver.dataset)
            _write_checkpoint(status="active", n_pending=0)
            registry.save_dataset(exp_id, driver.dataset.x_data[0],
                                   driver.dataset.y_data[0], set_completed=False)

        results, best_x_dict, best_y = _extract_results(driver, param_specs, fixed_context)
        best_cost = -best_y if best_y is not None else None

        data_points = []
        for i, r in enumerate(results):
            pt_lbl = (("seed-{}".format(i + 1) if use_prior else "init-{}".format(i + 1))
                      if i < n_warmup else "bo-{}".format(i - n_warmup + 1))
            x_dict = r.get("x") or {}
            dp_params = {**fixed_context, **x_dict}
            dp = {"label": pt_lbl, "params": dp_params,
                  "cost": -r["y"] if r.get("y") is not None else None}
            if best_x_dict and all(dp_params.get(k) == best_x_dict.get(k) for k in opt_var_names):
                dp["is_best"] = True
            data_points.append(dp)

        if best_cost is not None:
            best_vals = ", ".join("{}={:.4g}".format(k, v)
                                  for k, v in sorted(best_x_dict.items())
                                  if k in opt_var_names)
            print("  Best: [{}], cost=${:.2f}".format(best_vals, best_cost))

        registry.save_dataset(exp_id, driver.dataset.x_data[0], driver.dataset.y_data[0])
        registry.update_entry(
            exp_id,
            best_x=best_x_dict,
            best_y=best_y,
            n_samples=len([r for r in results if r.get("y") is not None]),
        )
        return {"data_points": data_points, "reuse_note": None, "error": None}

    except Exception as exc:
        import traceback
        registry.update_entry(exp_id, run_status="error")
        return {"data_points": [], "reuse_note": None,
                "error": "Optimization error: {}\n{}".format(exc, traceback.format_exc())}


def _run_surrogate_eval_step(step: dict) -> dict:
    """Recreate surrogate from saved registry data and query at a point."""
    eval_point = dict(step.get("fixed_context") or {})
    label      = step.get("label") or "surrogate-eval"

    print("  Querying surrogate at: {}".format(
        ", ".join("{}={}".format(k, v) for k, v in sorted(eval_point.items()))))

    try:
        all_entries = registry.list_entries()
    except Exception as exc:
        return {"data_points": [], "reuse_note": None,
                "error": "Cannot list experiments: {}".format(exc)}

    candidates = [
        e for e in all_entries
        if e.get("run_status") == "completed"
        and e.get("experiment_type") == "optimization"
        and all(eval_point.get(k) == v
                for k, v in (e.get("fixed_context") or {}).items())
    ]
    if not candidates:
        return {"data_points": [], "reuse_note": None,
                "error": (
                    "No completed optimization experiment found whose fixed_context "
                    "is consistent with the evaluation point {}. "
                    "Run a run_optimization step first.".format(eval_point)
                )}

    exp           = max(candidates, key=lambda e: e.get("created_at", ""))
    param_specs   = exp.get("param_specs", [])
    fixed_context = exp.get("fixed_context", {})

    print("  Using surrogate from experiment {} ({})".format(
        exp["id"][:8], exp.get("name", "?")))

    if not param_specs:
        return {"data_points": [], "reuse_note": None,
                "error": "Experiment {} has no param_specs.".format(exp["id"][:8])}

    x_row = []
    for spec in param_specs:
        name  = spec["name"]
        val   = eval_point.get(name)
        if val is None:
            return {"data_points": [], "reuse_note": None,
                    "error": "Evaluation point is missing value for param '{}'.".format(name)}
        ptype = spec["type"].lower()
        if ptype == "categorical":
            cats = spec.get("categories", [])
            if val not in cats:
                return {"data_points": [], "reuse_note": None,
                        "error": "Value '{}' not in categories {} for '{}'.".format(
                            val, cats, name)}
            x_row.append(float(cats.index(val)))
        else:
            x_row.append(float(val))

    try:
        data = registry.load_dataset(exp["id"])
    except FileNotFoundError as exc:
        return {"data_points": [], "reuse_note": None,
                "error": "Dataset not found for experiment {}: {}".format(exp["id"][:8], exc)}

    ac_params = build_ac_params(param_specs)
    formatter = build_task_formatter(param_specs, fixed_context,
                                     [_SESSION_MANAGER.machine_name])
    driver = ActiveLoopDriverHero(
        simulations=[None],
        params=ac_params,
        machine_names=[_SESSION_MANAGER.machine_name],
        output_field_path="y_data",
        surrogate="SMT_GP",
        acq_func="expected_improvement",
        blocking=False,
        task_formatter=formatter,
        hero_client=_SESSION_HERO,
    )
    driver.dataset.add_known_samples(data["x_data"], data["y_data"], 0)
    driver.surrogate.train(driver.dataset)

    x_query   = np.array([x_row])
    mean_y    = float(driver.surrogate.predict_values(x_query)[0][0])
    var_y     = float(driver.surrogate.predict_variances(x_query)[0][0])
    mean_cost = -mean_y
    std       = var_y ** 0.5

    std_str = " (±${:.2f} std)".format(std)
    print("  Surrogate prediction: cost=${:.2f}{}".format(mean_cost, std_str))

    dp = dict(eval_point)
    dp["label"]         = label
    dp["cost"]          = mean_cost
    dp["surrogate_std"] = std
    dp["is_surrogate"]  = True
    return {"data_points": [dp], "reuse_note": None, "error": None}


# ---------------------------------------------------------------------------
# 5. Graph nodes  (identical to rental_agent.py except search_registry)
# ---------------------------------------------------------------------------

def clarify(state):
    llm      = _get_llm()
    decision = llm.with_structured_output(ClarificationDecision)
    history  = state.get("conversation_history") or []
    prev_ctx = state.get("clarification_context") or ""

    messages = [SystemMessage(content=_CLARIFY_SYSTEM_PROMPT)]
    for turn in history:
        messages.append(HumanMessage(content=turn["request"]))
        messages.append(SystemMessage(content="Previous response summary: " + turn["response"][:300]))
    messages.append(HumanMessage(content=state["user_request"]))
    if prev_ctx:
        messages.append(SystemMessage(content="Prior clarification already provided: " + prev_ctx))

    try:
        result = decision.invoke(messages)
    except Exception:
        return {"clarification_context": prev_ctx or None}

    if not result.needs_clarification or not result.questions:
        return {"clarification_context": prev_ctx or None}

    print("\n--- Clarifying questions ---")
    for i, q in enumerate(result.questions, 1):
        print("  {}. {}".format(i, q))
    print()
    try:
        answers = input("Your answers: ").strip()
    except (EOFError, KeyboardInterrupt):
        answers = ""

    ctx = prev_ctx
    if answers:
        ctx = (ctx + "\n" if ctx else "") + "User clarifications: " + answers
    return {"clarification_context": ctx or None}


def plan(state):
    llm      = _get_llm()
    plan_llm = llm.with_structured_output(ResearchPlan)

    reset = {
        "plan_steps": [], "plan_reasoning": None, "plan_description": None,
        "completed_steps": [], "accumulated_results": [], "reuse_notes": [],
        "error": None, "response": None, "plan_feedback": None,
    }

    registry_summary = _fetch_registry_summary()
    history       = state.get("conversation_history") or []
    clarification = state.get("clarification_context") or ""
    feedback      = state.get("plan_feedback") or ""

    human_parts = [state["user_request"]]
    if clarification:
        human_parts.append("\nClarification provided: " + clarification)
    if feedback:
        human_parts.append("\nPlan revision requested: " + feedback)
    human_parts.append("\n\n" + registry_summary)

    messages = [SystemMessage(content=_PLAN_SYSTEM_PROMPT)]
    for turn in history:
        messages.append(HumanMessage(content=turn["request"]))
        messages.append(SystemMessage(content="Previous response summary: " + turn["response"][:400]))
    messages.append(HumanMessage(content="\n".join(human_parts)))

    try:
        result = plan_llm.invoke(messages)
        pd     = result.model_dump()
        steps  = pd["steps"]

        print("\nPlan: {} ({} steps)".format(pd["description"], len(steps)))
        for i, s in enumerate(steps, 1):
            lbl = " ({})".format(s["label"]) if s.get("label") else ""
            detail = ""
            if s["tool"] == "run_simulation":
                ctx = s.get("fixed_context") or {}
                detail = "  → {}".format(
                    "  ".join("{}={}".format(k, v) for k, v in sorted(ctx.items())))
            elif s["tool"] == "run_exploration":
                detail = "  → {} LHS samples".format(s.get("n_exploration_samples"))
            elif s["tool"] == "run_optimization":
                specs     = s.get("param_specs") or []
                ctx       = s.get("fixed_context") or {}
                n_bat_d   = s.get("n_bo_batches") or 1
                n_par_d   = s.get("n_parallel_per_batch") or 1
                fixed_str = "  ".join("{}={}".format(k, v) for k, v in sorted(ctx.items()))
                detail = "  → BO over [{}]: fixed=({}) (init={}, {}×{} BO = {} evals)".format(
                    _fmt_specs(specs), fixed_str,
                    s.get("n_init_samples"), n_bat_d, n_par_d, n_bat_d * n_par_d)
            elif s["tool"] == "evaluate_surrogate":
                ctx    = s.get("fixed_context") or {}
                detail = "  → surrogate at ({})".format(
                    "  ".join("{}={}".format(k, v) for k, v in sorted(ctx.items())))
            print("  Step {}: [{}]{} — {}{}".format(i, s["tool"], lbl, s["purpose"], detail))
        print("  Reasoning:", pd["reasoning"][:200], "...")

        return {**reset,
                "plan_steps": steps,
                "plan_reasoning": pd["reasoning"],
                "plan_description": pd["description"],
                "status": "plan_ready"}
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print("\n[ERROR] Planning failed: {}\n{}".format(exc, tb))
        return {**reset, "status": "error",
                "error": "Planning failed: {}\n{}".format(exc, tb)}


def approve_direction(state):
    if state.get("status") == "error":
        return {}
    steps = state.get("plan_steps") or []
    print("\n" + "=" * 70)
    print("Proposed Research Direction: {}".format(state.get("plan_description", "")))
    print("=" * 70)
    print("Reasoning: {}\n".format(state.get("plan_reasoning", "")))
    for i, s in enumerate(steps, 1):
        lbl = " ({})".format(s["label"]) if s.get("label") else ""
        print("  Step {}: [{}]{} — {}".format(i, s["tool"], lbl, s["purpose"]))
    print()
    try:
        user_input = input("Does this research direction look right? [yes / feedback to revise]: ").strip()
    except (EOFError, KeyboardInterrupt):
        user_input = "yes"
    if user_input.lower() in _APPROVE_YES:
        print("Direction approved — searching registry...\n")
        return {"status": "direction_approved", "plan_feedback": None}
    else:
        print("Revising plan...\n")
        return {"status": "plan_ready", "plan_feedback": user_input}


def search_registry(state):
    if state.get("status") == "error":
        return {}
    steps = list(state.get("plan_steps") or [])
    try:
        all_entries = registry.list_entries()
    except Exception as exc:
        print("Warning: registry search failed ({}); assuming no reuse.".format(exc))
        all_entries = []

    def _param_key(specs):
        return sorted((s.get("name",""), s.get("type","")) for s in specs)

    annotated = []
    for step in steps:
        step = dict(step)
        step["_found"] = None
        fixed_ctx   = step.get("fixed_context") or {}
        step_specs  = step.get("param_specs") or []
        step_pk     = _param_key(
            [ps.model_dump() if hasattr(ps, "model_dump") else ps for ps in step_specs]
        )
        found_list = []
        for e in all_entries:
            if e.get("run_status") != "completed":
                continue
            ctx = e.get("fixed_context") or {}
            if step["tool"] == "run_simulation":
                if ctx == fixed_ctx:
                    step["_found"] = e
                    break
            elif step["tool"] == "run_optimization":
                if e.get("experiment_type") != "optimization":
                    continue
                if ctx != fixed_ctx:
                    continue
                if _param_key(e.get("param_specs") or []) == step_pk:
                    found_list.append(e)
        if step["tool"] == "run_optimization" and step["_found"] is None:
            step["_found"] = found_list
        annotated.append(step)
    return {"plan_steps": annotated, "status": "searched"}


def negotiate_reuse(state):
    if state.get("status") == "error":
        return {}

    steps = list(state.get("plan_steps") or [])

    ws_indices = [
        i for i, s in enumerate(steps)
        if s.get("tool") == "run_optimization" and s.get("_found")
    ]
    if not ws_indices:
        return {}

    print("\n" + "=" * 70)
    print("Warm-Start Data Available — Reuse Options")
    print("=" * 70)
    print("Prior optimization data was found for the following step(s).")
    print("LHS initialization will be SKIPPED; existing samples seed the surrogate.")
    print("You can adjust the number of additional BO steps, or run fresh.\n")

    for i in ws_indices:
        s          = steps[i]
        prior_list = s.get("_found") or []
        total_n    = sum(pe.get("n_samples", 0) or 0 for pe in prior_list)
        lbl       = " ({})".format(s["label"]) if s.get("label") else ""
        specs     = s.get("param_specs") or []
        ctx       = s.get("fixed_context") or {}
        fixed_str = "  ".join("{}={}".format(k, v) for k, v in sorted(ctx.items()))
        print("  Step {}{}: opt=[{}]  fixed=({})".format(
            i + 1, lbl, _fmt_specs(specs), fixed_str))
        for pe in prior_list:
            p_n    = pe.get("n_samples", "?")
            p_best = pe.get("best_y")
            p_best_str = " best=${:.0f}".format(-p_best) if p_best is not None else ""
            print("    Prior data : [{}] {} samples{}".format(pe["id"][:8], p_n, p_best_str))
        n_bat_p = s.get("n_bo_batches") or 1
        n_par_p = s.get("n_parallel_per_batch") or 1
        bo_mode_p = "{}×{} parallel".format(n_bat_p, n_par_p) if n_par_p > 1 else "{}×1 sequential".format(n_bat_p)
        print("    Planned    : {} LHS init (auto-skipped) + {} BO evals ({})".format(
            s.get("n_init_samples") or 3, n_bat_p * n_par_p, bo_mode_p))
        print("    Total prior samples available to seed: {} (exact count after dedup may be lower)".format(total_n))
        print()

    print("Options:")
    print("  Enter      — keep planned BO steps (LHS skipped automatically)")
    print("  N          — 1 batch of N parallel BO evals  (e.g. '3')")
    print("  MxN        — M serial batches of N parallel  (e.g. '2 batches of 4')")
    print("  fresh      — discard prior data and run from scratch with LHS")
    if len(ws_indices) > 1:
        print("  Per-step   — e.g. 'step 2: 3 samples, step 3: fresh'")
    print()

    try:
        user_input = input("Reuse preference (or Enter to keep defaults): ").strip()
    except (EOFError, KeyboardInterrupt):
        user_input = ""

    if not user_input or user_input.lower() in _APPROVE_YES:
        for i in ws_indices:
            steps[i] = dict(steps[i])
            steps[i]["n_init_samples"] = 0
        return {"plan_steps": steps}

    context_lines = ["Warm-start optimization steps:"]
    for i in ws_indices:
        s       = steps[i]
        total_n = sum(pe.get("n_samples", 0) or 0 for pe in (s.get("_found") or []))
        n_bat_c = s.get("n_bo_batches") or 1
        n_par_c = s.get("n_parallel_per_batch") or 1
        context_lines.append(
            "  Step {}: {} prior samples, planned {} BO evals ({} batches × {} parallel)".format(
                i + 1, total_n, n_bat_c * n_par_c, n_bat_c, n_par_c)
        )
    context_lines += ["", 'User response: "{}"'.format(user_input)]

    try:
        llm    = _get_llm()
        parser = llm.with_structured_output(ReusePatchResult)
        result = parser.invoke([
            SystemMessage(content=_NEGOTIATE_REUSE_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(context_lines)),
        ])
        patches = {p.step_index: p for p in result.patches}
    except Exception as exc:
        print("Warning: could not parse reuse preferences ({}); keeping defaults.".format(exc))
        return {"plan_steps": steps}

    updated = []
    for i, s in enumerate(steps):
        s = dict(s)
        patch = patches.get(i + 1)
        if patch:
            if not patch.use_prior_data:
                s["_found"]     = []
                s["_fresh_run"] = True
                print("  Step {}: will run FRESH (prior data discarded)".format(i + 1))
            else:
                if patch.n_bo_batches is not None:
                    s["n_bo_batches"] = patch.n_bo_batches
                if patch.n_parallel_per_batch is not None:
                    s["n_parallel_per_batch"] = patch.n_parallel_per_batch
                s["n_init_samples"] = 0
                n_bat_u = s.get("n_bo_batches") or 1
                n_par_u = s.get("n_parallel_per_batch") or 1
                s["purpose"] = s["purpose"].rstrip() + "  [UPDATED: warm-start, {0}×{1} BO = {2} evals]".format(
                    n_bat_u, n_par_u, n_bat_u * n_par_u)
                print("  Step {}: warm-start, {} BO evals ({} batches × {} parallel)".format(
                    i + 1, n_bat_u * n_par_u, n_bat_u, n_par_u))
        updated.append(s)

    return {"plan_steps": updated}


def approve_concrete(state):
    if state.get("status") == "error":
        return {}
    steps = state.get("plan_steps") or []
    print("\n" + "=" * 70)
    print("Concrete Execution Plan: {}".format(state.get("plan_description", "")))
    print("=" * 70)
    for i, s in enumerate(steps, 1):
        lbl   = " ({})".format(s["label"]) if s.get("label") else ""
        found = s.get("_found")
        if s["tool"] == "evaluate_surrogate":
            status_str = "  → SURROGATE PREDICTION (no simulation or HPC)"
        elif s["tool"] == "run_simulation":
            if found and not isinstance(found, list):
                short_id  = found["id"][:8]
                best_y    = found.get("best_y")
                best_str  = " best=${:.0f}".format(-best_y) if best_y is not None else ""
                status_str = "  → REUSE [{}]{} — no new simulation".format(short_id, best_str)
            else:
                status_str = "  → RUN FRESH"
        elif s["tool"] == "run_optimization":
            prior_list = found if isinstance(found, list) else []
            if prior_list:
                prior_lines = []
                for pe in prior_list:
                    p_n    = pe.get("n_samples", "?")
                    p_best = pe.get("best_y")
                    p_best_str = " best=${:.0f}".format(-p_best) if p_best is not None else ""
                    prior_lines.append("[{id}] {n} samples{best}".format(
                        id=pe["id"][:8], n=p_n, best=p_best_str))
                status_str = (
                    "  → AUTO WARM-START from prior data:\n"
                    + "\n".join("           " + line for line in prior_lines)
                )
            else:
                status_str = "  → RUN FRESH (no prior optimization data — LHS warm-up)"
        else:
            status_str = "  → RUN FRESH"
        print("  Step {}: [{}]{} — {}".format(i, s["tool"], lbl, s["purpose"]))
        if s["tool"] == "run_optimization":
            n_init_raw = s.get("n_init_samples")
            n_init_d = 3 if n_init_raw is None else int(n_init_raw)
            n_bat_d  = s.get("n_bo_batches") or 1
            n_par_d  = s.get("n_parallel_per_batch") or 1
            has_ws   = bool(s.get("_found"))
            init_str = "0 (auto-skipped, warm-start)" if has_ws else str(n_init_d)
            bo_mode  = "{}×{} parallel".format(n_bat_d, n_par_d) if n_par_d > 1 else "{}×1 sequential".format(n_bat_d)
            print("           init={}  BO={} evals ({})".format(init_str, n_bat_d * n_par_d, bo_mode))
        print("         " + status_str)
    print()
    try:
        user_input = input("Approve execution? [yes / feedback to revise]: ").strip()
    except (EOFError, KeyboardInterrupt):
        user_input = "yes"
    if user_input.lower() in _APPROVE_YES:
        print("Approved — starting execution...\n")
        return {"status": "approved", "plan_feedback": None}
    else:
        print("Revising plan...\n")
        return {"status": "plan_ready", "plan_feedback": user_input}


def execute_step(state):
    steps = list(state.get("plan_steps") or [])
    if not steps:
        return {"status": "done"}

    step        = steps.pop(0)
    done_so_far = len(state.get("completed_steps") or [])
    total       = done_so_far + 1 + len(steps)

    print("\n[Step {}/{}] [{}] ({}) — {}".format(
        done_so_far + 1, total,
        step["tool"], step.get("label", ""), step["purpose"]))

    try:
        if step["tool"] == "run_simulation":
            result = _run_simulation_step(step)
        elif step["tool"] == "run_exploration":
            result = _run_exploration_step(step)
        elif step["tool"] == "run_optimization":
            result = _run_optimization_step(step)
        elif step["tool"] == "evaluate_surrogate":
            result = _run_surrogate_eval_step(step)
        else:
            result = {"data_points": [], "reuse_note": None,
                      "error": "Unknown tool: {}".format(step["tool"])}
    except Exception as exc:
        import traceback
        result = {"data_points": [], "reuse_note": None,
                  "error": "{}\n{}".format(exc, traceback.format_exc())}

    completed   = list(state.get("completed_steps") or [])
    completed.append({"step": step, "result": result})
    accumulated = list(state.get("accumulated_results") or [])
    accumulated.extend(result.get("data_points") or [])
    reuse_notes = list(state.get("reuse_notes") or [])
    if result.get("reuse_note"):
        reuse_notes.append(result["reuse_note"])

    if result.get("error"):
        return {"plan_steps": steps, "completed_steps": completed,
                "accumulated_results": accumulated, "reuse_notes": reuse_notes,
                "status": "error", "error": result["error"]}

    return {"plan_steps": steps, "completed_steps": completed,
            "accumulated_results": accumulated, "reuse_notes": reuse_notes,
            "status": "executing" if steps else "done", "error": None}


def synthesize_and_explain(state):
    accumulated     = state.get("accumulated_results") or []
    completed_steps = state.get("completed_steps") or []
    reuse_notes     = state.get("reuse_notes") or []
    user_req        = state.get("user_request", "")

    if state.get("status") == "error" and not accumulated and not completed_steps:
        err = state.get("error")
        if err:
            print("\n[ERROR] {}".format(err))
        return {}

    lines = [
        "USER REQUEST: {}".format(user_req), "",
        "RESEARCH PLAN:",
        "  Description: {}".format(state.get("plan_description", "?")),
        "  Reasoning  : {}".format(state.get("plan_reasoning", "")), "",
    ]
    if reuse_notes:
        lines.append("CACHE NOTES:")
        for note in reuse_notes:
            lines.append("  - " + note)
        lines.append("")

    lines.append("STEPS EXECUTED ({} total):".format(len(completed_steps)))
    for i, entry in enumerate(completed_steps, 1):
        s = entry["step"]
        r = entry["result"]
        lines.append("  Step {}: [{}] ({}) — {}".format(
            i, s["tool"], s.get("label", ""), s["purpose"]))
        if r.get("error"):
            lines.append("    ERROR: {}".format(str(r["error"])[:200]))
    lines.append("")

    lines.append("ALL RESULTS ({} data points):".format(len(accumulated)))
    for r in accumulated:
        cost_str   = "${:,.2f}".format(r["cost"]) if r.get("cost") is not None else "FAILED"
        if r.get("is_surrogate"):
            std = r.get("surrogate_std")
            cost_str += " ±${:.2f} (surrogate)".format(std) if std is not None else " (surrogate)"
        best_marker = " [BEST]" if r.get("is_best") else ""
        lines.append("  {} ({} | {} | {} EVs | SOC {}): {}{}".format(
            r.get("label", "?"),
            r.get("utility_rate", "?"), r.get("storage", "?"),
            r.get("number_of_daily_evs", "?"), r.get("return_soc", "?"),
            cost_str, best_marker))

    context_msg = "\n".join(lines)
    print("\nGenerating expert explanation...")
    try:
        llm      = _get_llm()
        response = llm.invoke([
            SystemMessage(content=_EXPLAIN_SYSTEM_PROMPT),
            HumanMessage(content=context_msg),
        ])
        explanation = response.content

        valid = [(r.get("label", "?"), r["cost"])
                 for r in accumulated if r.get("cost") is not None]

        header = ["=" * 70,
                  "Results ({} experiments)".format(len(accumulated))]
        for lbl, cost in valid:
            header.append("  {} -> ${:,.2f}".format(lbl, cost))
        if valid:
            best_lbl, best_cost = min(valid, key=lambda t: t[1])
            header.append("  BEST (lowest cost): {} -> ${:,.2f}".format(best_lbl, best_cost))
        header += ["=" * 70, "", "Expert Analysis", "-" * 70]
        full_response = "\n".join(header) + "\n" + explanation

        print("\n" + full_response)
        history = list(state.get("conversation_history") or [])
        history.append({"request": user_req, "response": full_response})
        return {"response": full_response, "conversation_history": history}
    except Exception as exc:
        import traceback
        fallback = "Explanation generation failed: {}\n{}".format(exc, traceback.format_exc())
        print(fallback)
        history = list(state.get("conversation_history") or [])
        history.append({"request": user_req, "response": fallback})
        return {"response": fallback, "conversation_history": history}


def ask_followup(state):
    print("\n" + "-" * 70)
    try:
        user_input = input("Follow-up question (or press Enter to quit): ").strip()
    except (EOFError, KeyboardInterrupt):
        user_input = ""
    if not user_input:
        print("Session ended.")
        return {"status": "done"}
    return {"user_request": user_input, "status": "continue",
            "clarification_context": None, "plan_feedback": None}


# ---------------------------------------------------------------------------
# 6. Routing  (identical to rental_agent.py)
# ---------------------------------------------------------------------------

def _route_after_plan(state):
    return "synthesize_and_explain" if state["status"] == "error" else "approve_direction"

def _route_after_approve_direction(state):
    if state["status"] == "direction_approved":
        return "search_registry"
    if state["status"] == "error":
        return "synthesize_and_explain"
    return "plan"

def _route_after_search_registry(state):
    if state.get("status") == "error":
        return "synthesize_and_explain"
    steps = state.get("plan_steps") or []
    has_warmstart = any(
        s.get("tool") == "run_optimization" and s.get("_found")
        for s in steps
    )
    return "negotiate_reuse" if has_warmstart else "approve_concrete"

def _route_after_approve_concrete(state):
    if state["status"] == "approved":
        return "execute_step"
    if state["status"] == "error":
        return "synthesize_and_explain"
    return "plan"

def _route_after_execute(state):
    return "execute_step" if state["status"] == "executing" else "synthesize_and_explain"

def _route_after_followup(state):
    return "clarify" if state["status"] == "continue" else END


# ---------------------------------------------------------------------------
# 7. Graph assembly  (identical to rental_agent.py)
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("clarify",                clarify)
    builder.add_node("plan",                   plan)
    builder.add_node("approve_direction",      approve_direction)
    builder.add_node("search_registry",        search_registry)
    builder.add_node("negotiate_reuse",        negotiate_reuse)
    builder.add_node("approve_concrete",       approve_concrete)
    builder.add_node("execute_step",           execute_step)
    builder.add_node("synthesize_and_explain", synthesize_and_explain)
    builder.add_node("ask_followup",           ask_followup)

    builder.set_entry_point("clarify")
    builder.add_edge("clarify",                    "plan")
    builder.add_conditional_edges("plan",              _route_after_plan)
    builder.add_conditional_edges("approve_direction", _route_after_approve_direction)
    builder.add_conditional_edges("search_registry",   _route_after_search_registry)
    builder.add_edge("negotiate_reuse",               "approve_concrete")
    builder.add_conditional_edges("approve_concrete",  _route_after_approve_concrete)
    builder.add_conditional_edges("execute_step",      _route_after_execute)
    builder.add_edge("synthesize_and_explain",         "ask_followup")
    builder.add_conditional_edges("ask_followup",      _route_after_followup)
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 8. Public API
# ---------------------------------------------------------------------------

def run_agent(
    user_request: str,
    chat_id: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
    prior_history: Optional[list] = None,
):
    """
    Start an interactive co-scientist session for rental car electrification.

    Parameters
    ----------
    user_request    : The user's goal / research question.
    chat_id         : If set, checkpoint writes are enabled.  Supplied by
                      co_scientist.py when launching a managed session.
    checkpoint_file : Path to write checkpoint JSON.  Required when chat_id
                      is provided.
    prior_history   : conversation_history from a previous run (for follow-up
                      sessions).
    """
    global _CHAT_ID, _CHECKPOINT_FILE, _CHECKPOINT_STATE
    global _SESSION_HERO, _SESSION_MANAGER

    if chat_id:
        _CHAT_ID = chat_id
        _CHECKPOINT_FILE = checkpoint_file
        _CHECKPOINT_STATE = {
            "chat_id": chat_id,
            "name": user_request[:80],
            "status": "active",
            "user_request": user_request,
            "n_pending": 0,
            "best_y": None,
            "best_x": None,
            "latest_run_id": None,
            "experiment_ids": [],
            "conversation_history": prior_history or [],
        }
        _write_checkpoint()
        session_name = "co-sci-{}".format(chat_id[:8])
        print("[agent] Session: {}  →  To kill: tmux kill-session -t {}".format(
            session_name, session_name))

    # Initialize session-scoped LocalHero client and inline manager.
    run_uuid = chat_id or _uuid_module.uuid4().hex[:8]
    hero_db  = str(_AGENT_DIR / "hero_db_{}.json".format(run_uuid))
    _SESSION_HERO    = LocalHeroClient(
        db_path=hero_db,
        queue_name="jobs",
        application_id="rental_agent_HPC_onsite",
    )
    _SESSION_MANAGER = create_manager(
        machine_name=MACHINE_NAME,
        hero_client=_SESSION_HERO,
        work_dir=str(_AGENT_DIR),
    )

    # Flush any in-flight tasks left in the hero DB from a previous session
    # with the same chat_id.  This must happen BEFORE LangGraph resumes and
    # any step function creates a new ActiveLoopDriverHero, because
    # HeroDataset.__init__ calls clear_hero_queue() which would otherwise
    # wipe the old 'running' tasks before the manager can reconnect to them.
    # For a fresh session the hero queue does not yet exist and run_until_done
    # creates it and exits immediately (n_ready + n_running == 0).
    try:
        _SESSION_MANAGER.run_until_done(i_fidelity=0)
    except Exception as exc:
        print("[startup] Hero queue flush: {}".format(exc))

    initial_state = {
        "user_request":          user_request,
        "conversation_history":  list(prior_history or []),
        "clarification_context": None,
        "plan_feedback":         None,
        "plan_steps":            [],
        "plan_reasoning":        None,
        "plan_description":      None,
        "completed_steps":       [],
        "accumulated_results":   [],
        "reuse_notes":           [],
        "status":                "pending",
        "error":                 None,
        "response":              None,
    }
    print("Request: {}\n".format(user_request))
    try:
        if chat_id:
            from langgraph.checkpoint.sqlite import SqliteSaver
            import chat_registry as _chat_registry
            sqlite_db = str(
                _chat_registry.checkpoint_path(chat_id).with_suffix(".db")
            )
            with SqliteSaver.from_conn_string(sqlite_db) as checkpointer:
                graph  = build_graph(checkpointer=checkpointer)
                config = {"configurable": {"thread_id": chat_id}}
                snap   = graph.get_state(config)
                if snap.next:
                    print("[controller] Resuming interrupted session from checkpoint...")
                    final = graph.invoke(None, config=config)
                else:
                    final = graph.invoke(initial_state, config=config)
        else:
            graph = build_graph()
            final = graph.invoke(initial_state)
    except Exception as exc:
        _write_checkpoint(status="error", error=str(exc))
        raise

    conv_hist = final.get("conversation_history", [])
    if final.get("status") == "error":
        _write_checkpoint(status="error", conversation_history=conv_hist)
        print("\n[ERROR] {}".format(final.get("error")))
    else:
        best_y = None
        for r in reversed(final.get("accumulated_results", [])):
            if r.get("best_y") is not None:
                best_y = r["best_y"]
                break
        _write_checkpoint(
            status="completed",
            n_pending=0,
            best_y=best_y,
            conversation_history=conv_hist,
        )
    return final


# ---------------------------------------------------------------------------
# 9. CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    examples = [
        "What storage percentage minimizes daily cost for a facility with 5000 EVs/day "
        "using an Aggressive utility rate?",

        "Compare Moderate vs Aggressive utility rates for a medium-demand facility "
        "(1000 EVs/day, SOC 35). Which is cheaper and by how much?",

        "Survey the full parameter space with LHS sampling and explain which parameters "
        "have the biggest impact on cost.",

        "Conduct a parallel Bayesian optimization (3 initial samples, 1 batch of 5 parallel BO samples) "
        "to find the cost-minimizing demand (number of daily EVs) with fixed state of charge = 30, "
        "fixed storage=40 percent, and fixed utility rate = Aggressive.",

        "Survey the parameter space holding utility_rate=moderate fixed and the other variables varying. "
        "Use 5 initial samples and 1 batch of 5 BO samples. Then use the surrogate to evaluate the point "
        "utility_rate=moderate, storage=50, number_of_daily_evs=2000, return_soc=40.",

        "Conduct a Bayesian optimization (4 initial samples + 1 batch of 4 BO samples) to find the "
        "cost-minimizing demand and state of charge (2 variable optimization) with fixed storage=20 "
        "percent and fixed utility rate=moderate.",
    ]

    parser = argparse.ArgumentParser(
        description="Rental-car electrification co-scientist agent (HPC_onsite variant)",
        add_help=False,
    )
    parser.add_argument("--chat-id",    default=None, metavar="UUID",
                        help="Unique ID for this chat session (set by co_scientist.py).")
    parser.add_argument("--checkpoint", default=None, metavar="PATH",
                        help="Path to the checkpoint JSON for this session.")
    parser.add_argument("--resume",     action="store_true",
                        help="Load prior_history from the checkpoint file before starting.")
    parser.add_argument("goal",  nargs="?", default=None,
                        help="Research goal.  Pass a single digit to pick an example.")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    parts    = []
    if args.goal:
        parts.append(args.goal)
    if args.extra:
        parts.extend(args.extra)
    arg_goal = " ".join(parts).strip()

    prior_hist = []
    if args.resume and args.checkpoint:
        try:
            import chat_registry
            ckpt = chat_registry.read_checkpoint(args.checkpoint)
            prior_hist = ckpt.get("conversation_history", [])
            if not arg_goal:
                arg_goal = ckpt.get("user_request", "")
            if prior_hist:
                print("[Resuming with {} prior conversation turn(s)]".format(len(prior_hist)))
        except Exception as exc:
            print("[checkpoint] Warning: could not load prior history: {}".format(exc))

    if arg_goal:
        if arg_goal.isdigit() and 1 <= int(arg_goal) <= len(examples):
            req = examples[int(arg_goal) - 1]
            print("Using example {}: {}\n".format(arg_goal, req))
        else:
            req = arg_goal
    else:
        print("Example prompts:")
        for i, ex in enumerate(examples, 1):
            print("  [{}] {}".format(i, ex))
        print()
        user_input = input(
            "Enter prompt (or 1–{} for an example): ".format(len(examples))
        ).strip()
        if user_input.isdigit() and 1 <= int(user_input) <= len(examples):
            req = examples[int(user_input) - 1]
        elif user_input:
            req = user_input
        else:
            req = examples[0]
        print("Using: {}\n".format(req))

    run_agent(
        req,
        chat_id=args.chat_id,
        checkpoint_file=args.checkpoint,
        prior_history=prior_hist or None,
    )
