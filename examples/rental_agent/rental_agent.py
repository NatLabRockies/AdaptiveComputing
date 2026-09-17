#!/usr/bin/env python3
"""
rental_agent.py — Rental Car Electrification Co-Scientist Agent
================================================================
An interactive LangGraph agent that acts as a research partner for
optimizing EV fleet electrification at an airport rental car facility.

The user describes a goal in natural language; the agent:
  1. (Optionally) asks targeted clarifying questions
  2. Proposes a high-level multi-step research plan
  3. User approves the *research direction* (or requests a revision)
  4. Agent searches the registry — shows REUSE vs FRESH RUN per step
  5. User approves the *concrete execution plan*
  6. Executes each step — run_simulation, run_exploration, or
     run_optimization — reusing cached experiments automatically
  7. Synthesizes an expert explanation of all results
  8. Asks for follow-up questions and loops until the user is done

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

All simulation / optimisation is delegated to the AC MCP server.

Usage
-----
    python rental_agent.py
    python rental_agent.py "What storage minimizes cost for 5000 EVs/day?"
    python rental_agent.py "Compare Moderate vs Aggressive utility rates across all storage options."
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import List, Literal, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from adaptive_computing.utils import load_env_file
load_env_file(os.path.join(_AGENT_DIR, "env_vars.txt"))

# ---------------------------------------------------------------------------
# MCP client helpers  (identical pattern to snn_agent.py)
# ---------------------------------------------------------------------------
_AC_MCP_URL      = os.environ.get("AC_MCP_URL", "http://localhost:8765/mcp")
_HPC_CONFIG_PATH = os.path.join(_AGENT_DIR, "hpc_config.py")
_OUTPUT_LABEL    = "Daily cost (USD)"
_AC_MCP_STORAGE_DIR  = _AGENT_DIR
_AC_MCP_START_SCRIPT = os.environ.get(
    "AC_MCP_START_SCRIPT",
    os.path.expanduser("~/AdaptiveComputing/ac_mcp/start_server.sh"),
)
_MCP_SESSION_NAME = "ac_mcp_server"


def _ensure_server_running() -> None:
    """Start the AC MCP server in a tmux session if it is not reachable."""
    import urllib.error
    import urllib.request
    from adaptive_computing.hpc.local_launcher import ensure_command_running

    url = _AC_MCP_URL.rstrip("/") + "/"  # health-check the root endpoint

    def _server_responds(timeout):
        try:
            urllib.request.urlopen(url, timeout=timeout)
            return True
        except urllib.error.HTTPError:
            return True   # server replied with HTTP status — it's up
        except (urllib.error.URLError, OSError):
            return False

    if _server_responds(3):
        return  # already running

    if not os.path.exists(_AC_MCP_START_SCRIPT):
        print(f"WARNING: AC MCP start script not found at {_AC_MCP_START_SCRIPT}")
        print("Start the server manually, then retry.")
        return

    print(f"Starting AC MCP server in tmux session '{_MCP_SESSION_NAME}'...")
    port = _AC_MCP_URL.split(":")[-1].split("/")[0]
    cmd = f"bash {_AC_MCP_START_SCRIPT!r} {_AC_MCP_STORAGE_DIR!r} {port}"
    ensure_command_running(
        session_name=_MCP_SESSION_NAME,
        command=cmd,
        log_file="/tmp/ac_mcp_server.log",
    )

    for _ in range(30):
        time.sleep(1)
        if _server_responds(2):
            print("AC MCP server ready.")
            return
    raise RuntimeError(
        "AC MCP server did not become ready after 30 s. "
        "Check logs with: tail -f /tmp/ac_mcp_server.log"
    )


def _call_tool(tool_name: str, **kwargs) -> dict:
    """Call one AC MCP tool synchronously via fastmcp and return its result."""
    from fastmcp import Client
    import json as _json

    async def _run():
        async with Client(_AC_MCP_URL) as client:
            return await client.call_tool(tool_name, kwargs)

    result = asyncio.run(_run())
    # FastMCP returns a CallToolResult with a .content list of TextContent.
    if hasattr(result, "content"):
        content_list = result.content
    elif isinstance(result, list):
        content_list = result
    else:
        content_list = []
    if content_list:
        raw = content_list[0]
        text = raw.text if hasattr(raw, "text") else str(raw)
        return _json.loads(text)
    return result


# ---------------------------------------------------------------------------
# Checkpoint context (populated when running under co_scientist.py)
# ---------------------------------------------------------------------------

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


def _poll_run(run_id: str, poll_interval: float = 15.0) -> dict:
    """Block until the run is completed or errored, printing progress."""
    _write_checkpoint(status="waiting", latest_run_id=run_id)
    while True:
        status  = _call_tool("get_run_status", run_id=run_id)
        n_done  = status.get("n_completed", 0)
        n_total = status.get("n_total", "?")
        state   = status.get("status", "running")
        best_y  = status.get("best_y")
        msg     = status.get("message", "")
        best_str  = " best=${:.2f}".format(-best_y) if best_y is not None else ""
        phase_str = " [{}]".format(msg) if msg else ""
        print("  [{}/{}]{}{} ({})".format(n_done, n_total, best_str, phase_str, state))
        n_pending = (n_total - n_done) if isinstance(n_total, int) else 0
        _write_checkpoint(
            status="waiting",
            n_pending=max(0, n_pending),
            best_y=best_y,
            latest_run_id=run_id,
        )
        if state in ("completed", "error"):
            _write_checkpoint(status="active", n_pending=0, best_y=best_y)
            return status
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Domain value sets
# ---------------------------------------------------------------------------
_UTILITY_RATES = ["Moderate", "Aggressive"]

# Default exploration space — used by _run_exploration_step when param_specs is empty.
_DEFAULT_EXPLORATION_SPECS = [
    {"name": "utility_rate",        "type": "categorical", "categories": _UTILITY_RATES},
    {"name": "storage",             "type": "continuous",  "min": 0.0,   "max": 100.0},
    {"name": "number_of_daily_evs", "type": "continuous",  "min": 10.0,  "max": 10000.0},
    {"name": "return_soc",          "type": "continuous",  "min": 25.0,  "max": 55.0},
]


from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# 1. Schemas
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
    # Each entry: {name, type, min/max or categories}.
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
    """LLM-parsed preference for one warm-start optimization step."""
    step_index:           int           = Field(..., description="1-based step index from the plan")
    use_prior_data:       bool          = Field(True,  description="True=warm-start from prior data; False=discard and run fresh with LHS")
    n_bo_batches:         Optional[int] = Field(None,  description="Override serial BO rounds; None=keep original planned value")
    n_parallel_per_batch: Optional[int] = Field(None,  description="Override parallel evaluations per round; None=keep original planned value")


class ReusePatchResult(BaseModel):
    """Parsed reuse preferences for all warm-start optimization steps."""
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
# 2. System prompts
# ---------------------------------------------------------------------------

_SIMULATOR_CONTEXT = """\
## Simulator Context
You are advising on a pre-built, fixed black-box rental car electrification model.
The following properties are FIXED and cannot be changed:

  Facility     : Airport rental car center with EV fleet
  Objective    : Minimize total daily energy cost (USD)
  Metric       : cost — total daily cost to charge the fleet

The model is a black box: the exact relationship between inputs and cost is
unknown and must be discovered through simulation and optimization.

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
Evaluate one fixed configuration.  Set in the PlanStep:
  fixed_context: dict with ALL parameter values, e.g.
    {"utility_rate": "Moderate", "storage": 50.0, "number_of_daily_evs": 1000.0, "return_soc": 40.0}
  label, purpose.

### run_exploration
Draw N LHS samples from the parameter space for a broad cost survey.  Set:
  n_exploration_samples (default 20), label, purpose.
  param_specs: list of parameters to sweep (omit or empty = sweep all 4).
  fixed_context: dict of any parameters to hold fixed during the sweep (may be empty).

### run_optimization
Bayesian-optimize any subset of parameters jointly.  Set:
  param_specs: list of FREE variables, each as:
    {"name": "storage",             "type": "continuous",  "min": 0.0,   "max": 100.0}
    {"name": "number_of_daily_evs", "type": "continuous",  "min": 10.0,  "max": 10000.0}
    {"name": "return_soc",          "type": "continuous",  "min": 25.0,  "max": 55.0}
    {"name": "utility_rate",        "type": "categorical", "categories": ["Moderate","Aggressive"]}
  fixed_context: dict of parameters held FIXED, e.g.
    {"utility_rate": "Moderate", "number_of_daily_evs": 1000.0}
  n_init_samples (default 3), n_bo_batches (default 1), n_parallel_per_batch (default 1).

  Total BO evaluations = n_bo_batches × n_parallel_per_batch.
  n_parallel_per_batch=1 is sequential (most sample-efficient).
  n_parallel_per_batch>1 is a parallel batch (faster wall-clock, less efficient).
  Ask the user when they specify a batch structure; otherwise default to 1×1.

### evaluate_surrogate
Query the trained surrogate from a prior run_optimization — no simulation needed.  Set:
  fixed_context: dict with ALL parameter values at the evaluation point.
  label, purpose.
Pair with a prior run_optimization whose param_specs are consistent.

## Creative Composition Examples
  Compare two configs          → 2 × run_simulation (different fixed_context each)
  Sweep utility rates          → run_simulation × 2 (fixed_context differs by utility_rate)
  Survey full space            → run_exploration (n_exploration_samples, empty param_specs)
  Optimize storage + SOC       → run_optimization(
                                   param_specs=[{storage,continuous,0,100},{return_soc,continuous,25,55}],
                                   fixed_context={"utility_rate":"Moderate","number_of_daily_evs":1000})
  Optimize everything          → run_optimization(param_specs=[all 4], fixed_context={})
  Baseline then optimize       → run_simulation(fixed_context={...,storage=0}) + run_optimization
  Multi-scenario study         → run_optimization × N (different fixed_context per run)
  Optimize then predict        → run_optimization + evaluate_surrogate

Set 'reasoning' with scientific rationale grounded in the parameter ranges.
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


# ---------------------------------------------------------------------------
# 3. LLM factory
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
    """Return a plain-text summary of the experiment registry."""
    try:
        result = _call_tool("list_experiments")
        experiments = result if isinstance(result, list) else result.get("experiments", [])
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


def _fmt_specs(specs: list) -> str:
    """Format a list of param_specs for display."""
    parts = []
    for ps in specs:
        if isinstance(ps, dict):
            name, ptype = ps.get("name", "?"), ps.get("type", "")
        else:
            name, ptype = ps.name, ps.type
            ps = ps.model_dump()
        if ptype in ("continuous", "ordered"):
            parts.append("{}∈[{},{}]".format(name, ps.get("min", "?"), ps.get("max", "?")))
        else:
            parts.append("{}∈{}".format(name, ps.get("categories", [])))
    return "  ".join(parts)


def _run_simulation_step(step: dict) -> dict:
    """Execute one run_simulation step via the AC MCP server."""
    fixed_ctx = dict(step.get("fixed_context") or {})
    label     = step.get("label") or "sim-{}".format(
        "-".join(str(v) for v in list(fixed_ctx.values())[:3]))

    print("  Submitting: {}".format(
        "  ".join("{}={}".format(k, v) for k, v in sorted(fixed_ctx.items()))))

    exp = _call_tool(
        "create_experiment",
        name=step.get("label") or "rental-car-simulation",
        description=step["purpose"],
        param_specs=[],
        fixed_context=fixed_ctx,
        output_label=_OUTPUT_LABEL,
        hpc_config_path=_HPC_CONFIG_PATH,
        experiment_type="evaluation",
    )
    exp_id = exp["experiment_id"]

    if exp.get("reused"):
        print("  (reusing completed experiment {})".format(exp_id[:8]))
        raw_best = exp.get("best_y")
        cost = -raw_best if raw_best is not None else None
        return {
            "data_points": [{"label": label, "params": fixed_ctx, "cost": cost}],
            "reuse_note": "Reused cached result for {}.".format(label),
            "error": None,
        }

    run    = _call_tool("run_evaluations", experiment_id=exp_id, jobs=[{}])
    run_id = run["run_id"]
    print("  run_id:", run_id)

    status  = _poll_run(run_id)
    if status["status"] == "error":
        return {"data_points": [], "reuse_note": None,
                "error": "Simulation error: {}".format(status.get("error", "?"))}

    results  = status.get("results", [])
    raw_cost = results[0]["y"] if results else None
    if raw_cost is None:
        return {"data_points": [], "reuse_note": None,
                "error": "Simulation returned no result — check Slurm logs in cases_agent/"}
    cost = -raw_cost

    return {
        "data_points": [{"label": label, "params": fixed_ctx, "cost": cost}],
        "reuse_note": None, "error": None,
    }


def _lhs_jobs(n_samples: int, seed: int = 42) -> list:
    """Generate Latin Hypercube Samples as explicit parameter dicts.

    Covers all 4 parameters: utility_rate (categorical), storage, number_of_daily_evs,
    return_soc.  Uses Python's built-in random so no extra dependencies are needed.
    """
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


def _run_exploration_step(step: dict) -> dict:
    """Execute one run_exploration step (LHS survey over the parameter space)."""
    n_samples   = int(step.get("n_exploration_samples") or 20)
    label       = step.get("label") or "exploration-{}".format(n_samples)
    # Use param_specs from step if provided; fall back to the full 4-param default.
    raw_specs   = step.get("param_specs") or []
    param_specs = ([ps.model_dump() if hasattr(ps, "model_dump") else dict(ps)
                    for ps in raw_specs]
                   if raw_specs else list(_DEFAULT_EXPLORATION_SPECS))
    fixed_ctx   = dict(step.get("fixed_context") or {})

    print("  Exploring: {} LHS samples  params=[{}]".format(
        n_samples, _fmt_specs(param_specs)))

    exp = _call_tool(
        "create_experiment",
        name=step.get("label") or "rental-car-exploration",
        description=step["purpose"],
        param_specs=param_specs,
        fixed_context=fixed_ctx,
        output_label=_OUTPUT_LABEL,
        hpc_config_path=_HPC_CONFIG_PATH,
        experiment_type="exploration",
    )
    exp_id = exp["experiment_id"]

    if exp.get("reused"):
        print("  (reusing completed exploration {})".format(exp_id[:8]))
        cached  = _call_tool("get_experiment", experiment_id=exp_id)
        results = cached.get("results", [])
        data_points = [
            {"label": "lhs-{}".format(i + 1),
             "params": {**fixed_ctx, **r["x"]},
             "cost": -r["y"]}
            for i, r in enumerate(results)
        ]
        return {"data_points": data_points,
                "reuse_note": "Reused cached exploration {}.".format(label),
                "error": None}

    run    = _call_tool("run_evaluations", experiment_id=exp_id,
                        jobs=_lhs_jobs(n_samples))
    run_id = run["run_id"]
    print("  run_id:", run_id)

    status = _poll_run(run_id, poll_interval=15.0)
    if status["status"] == "error":
        return {"data_points": [], "reuse_note": None,
                "error": "Exploration error: {}".format(status.get("error", "?"))}

    raw_results = status.get("results", [])
    data_points = []
    for i, r in enumerate(raw_results):
        data_points.append({
            "label":  "lhs-{}".format(i + 1),
            "params": {**fixed_ctx, **r.get("x", {})},
            "cost":   -r["y"],
        })

    return {"data_points": data_points, "reuse_note": None, "error": None}


def _run_surrogate_eval_step(step: dict) -> dict:
    """Query the trained surrogate from a prior optimization at a specific point."""
    eval_point = dict(step.get("fixed_context") or {})
    label      = step.get("label") or "surrogate-eval"

    print("  Querying surrogate at: {}".format(
        ", ".join("{}={}".format(k, v) for k, v in sorted(eval_point.items()))))

    # Find the most recent completed optimization whose fixed_context is
    # consistent with the evaluation point (all fixed keys match).
    try:
        raw_entries = _call_tool("list_experiments")
        all_entries = raw_entries if isinstance(raw_entries, list) else []
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

    exp = max(candidates, key=lambda e: e.get("created_at", ""))
    exp_id = exp["id"]
    print("  Using surrogate from experiment {} ({})".format(
        exp_id[:8], exp.get("name", "?")))

    # Encode the free-variable values into x_points using param_specs ordering.
    param_specs = exp.get("param_specs", [])
    if not param_specs:
        return {"data_points": [], "reuse_note": None,
                "error": "Experiment {} has no param_specs.".format(exp_id[:8])}

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
        result = _call_tool("predict", experiment_id=exp_id, x_points=[x_row])
    except Exception as exc:
        return {"data_points": [], "reuse_note": None,
                "error": "Surrogate prediction failed: {}".format(exc)}

    preds = result.get("predictions", [])
    if not preds:
        return {"data_points": [], "reuse_note": None,
                "error": "No predictions returned from surrogate."}

    pred       = preds[0]
    mean_y     = pred.get("mean")
    variance   = pred.get("variance")
    mean_cost  = -mean_y if mean_y is not None else None
    std        = variance ** 0.5 if variance is not None else None

    if mean_cost is not None:
        std_str = " (±${:.2f} std)".format(std) if std is not None else ""
        print("  Surrogate prediction: cost=${:.2f}{}".format(mean_cost, std_str))

    dp = {
        "label":        label,
        "params":       eval_point,
        "cost":         mean_cost,
        "surrogate_std": std,
        "is_surrogate": True,
    }
    return {"data_points": [dp], "reuse_note": None, "error": None}


def _run_optimization_step(step: dict) -> dict:
    """Execute one run_optimization step over any combination of parameters."""
    raw_specs     = step.get("param_specs") or []
    param_specs   = [ps.model_dump() if hasattr(ps, "model_dump") else dict(ps)
                     for ps in raw_specs]
    fixed_context = dict(step.get("fixed_context") or {})
    opt_var_names = [ps["name"] for ps in param_specs]

    if not param_specs:
        return {"data_points": [], "reuse_note": None,
                "error": "run_optimization: param_specs is empty — nothing to optimize."}

    n_init_raw = step.get("n_init_samples")
    n_init     = 3 if n_init_raw is None else int(n_init_raw)
    n_batches  = int(step.get("n_bo_batches") or 1)
    n_parallel = int(step.get("n_parallel_per_batch") or 1)
    n_steps    = n_batches * n_parallel
    blocking   = (n_parallel == 1)
    fixed_str  = "  ".join("{}={}".format(k, v) for k, v in sorted(fixed_context.items()))
    label      = step.get("label") or "opt-{}".format("+".join(opt_var_names))
    exp_name   = label

    print("  Optimizing [{}]: fixed=({})  (init={}, {}×{} BO = {} evals)".format(
        _fmt_specs(param_specs), fixed_str, n_init, n_batches, n_parallel, n_steps))

    # _found non-empty  → warm-start from prior data: create a NEW experiment so
    #                      run_manager can seed from the old one via find_reusable_data.
    # _fresh_run=True   → user explicitly discarded prior data: create a NEW experiment
    #                      and pass skip_warmstart=True so _opt_worker ignores ALL prior
    #                      data (including from other matching experiments).
    # otherwise         → normal force_new=False behaviour.
    has_warmstart = bool(step.get("_found"))      # non-empty list from search_registry
    fresh_run     = bool(step.get("_fresh_run"))  # set by negotiate_reuse when user says fresh
    force_new     = has_warmstart or fresh_run

    exp = _call_tool(
        "create_experiment",
        name=exp_name,
        description=step["purpose"],
        param_specs=param_specs,
        fixed_context=fixed_context,
        output_label=_OUTPUT_LABEL,
        hpc_config_path=_HPC_CONFIG_PATH,
        experiment_type="optimization",
        force_new=force_new,
    )
    exp_id = exp["experiment_id"]

    # Only skip new simulations when there's no warm-start data AND no fresh-run request.
    # In both other cases we fall through and call run_optimization.
    if exp.get("reused") and not has_warmstart and not fresh_run:
        print("  (reusing completed optimization {})".format(exp_id[:8]))
        best_x_raw = exp.get("best_x")
        raw_best   = exp.get("best_y")
        best_cost  = -raw_best if raw_best is not None else None
        if isinstance(best_x_raw, dict):
            best_x_dict = best_x_raw
        elif isinstance(best_x_raw, list):
            best_x_dict = dict(zip(opt_var_names, best_x_raw))
        else:
            best_x_dict = {}
        dp_params = {**fixed_context, **best_x_dict}
        return {
            "data_points": [{"label": label, "params": dp_params,
                              "cost": best_cost, "is_best": True}],
            "reuse_note": "Reused cached optimization for {}.".format(label),
            "error": None,
        }

    run = _call_tool(
        "run_optimization",
        experiment_id=exp_id,
        n_init_samples=n_init,
        n_steps=n_steps,
        acq_func="expected_improvement",
        blocking=blocking,
        skip_warmstart=fresh_run,
    )
    run_id = run["run_id"]
    print("  run_id:", run_id)

    status = _poll_run(run_id, poll_interval=10.0)
    if status["status"] == "error":
        return {"data_points": [], "reuse_note": None,
                "error": "Optimization error: {}".format(status.get("error", "?"))}

    raw_results = status.get("results", [])
    best_x_dict = status.get("best_x") or {}
    if isinstance(best_x_dict, list):
        best_x_dict = dict(zip(opt_var_names, best_x_dict))
    raw_best    = status.get("best_y")
    best_cost   = -raw_best if raw_best is not None else None
    n_warmup    = status.get("n_warmup", n_init)

    data_points = []
    for i, r in enumerate(raw_results):
        pt_lbl = (
            ("seed-{}".format(i + 1) if has_warmstart else "init-{}".format(i + 1))
            if i < n_warmup else "bo-{}".format(i - n_warmup + 1)
        )
        x_dict = r.get("x") or {}
        if isinstance(x_dict, list):
            x_dict = dict(zip(opt_var_names, x_dict))
        dp_params = {**fixed_context, **x_dict}
        dp = {"label": pt_lbl, "params": dp_params, "cost": -r["y"]}
        if all(dp_params.get(k) == best_x_dict.get(k) for k in opt_var_names):
            dp["is_best"] = True
        data_points.append(dp)

    if best_cost is None:
        return {"data_points": data_points, "reuse_note": None,
                "error": "All BO trials returned no result"}

    best_vals = "  ".join("{}={:.4g}".format(k, v)
                          for k, v in sorted(best_x_dict.items()) if k in opt_var_names)
    print("  Best: [{}], cost=${:.2f}".format(best_vals, best_cost))
    return {"data_points": data_points, "reuse_note": None, "error": None}


# ---------------------------------------------------------------------------
# 5. Graph nodes
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
            print("  Step {}: [{}]{} — {}".format(i, s["tool"], lbl, s["purpose"]))
            if s["tool"] == "run_simulation":
                ctx = s.get("fixed_context") or {}
                print("    fixed={}".format(
                    "  ".join("{}={}".format(k, v) for k, v in sorted(ctx.items()))))
            elif s["tool"] == "run_exploration":
                specs = s.get("param_specs") or []
                print("    {} LHS  params=[{}]".format(
                    s.get("n_exploration_samples") or 20,
                    _fmt_specs(specs) if specs else "all"))
            elif s["tool"] == "run_optimization":
                specs = s.get("param_specs") or []
                ctx   = s.get("fixed_context") or {}
                n_bat = s.get("n_bo_batches") or 1
                n_par = s.get("n_parallel_per_batch") or 1
                print("    opt=[{}]  fixed=({})  BO={}×{}={}".format(
                    _fmt_specs(specs),
                    "  ".join("{}={}".format(k, v) for k, v in sorted(ctx.items())),
                    n_bat, n_par, n_bat * n_par))
            elif s["tool"] == "evaluate_surrogate":
                ctx = s.get("fixed_context") or {}
                print("    surrogate at {}".format(
                    "  ".join("{}={}".format(k, v) for k, v in sorted(ctx.items()))))
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
        raw_entries = _call_tool("list_experiments")
        all_entries = raw_entries if isinstance(raw_entries, list) else []
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
        param_specs = step.get("param_specs") or []
        step_pk     = _param_key(
            [ps.model_dump() if hasattr(ps, "model_dump") else ps for ps in param_specs]
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
    """
    Node: when warm-start data exists for any optimization step, ask the user
    how they want to use it — reuse vs fresh, and how many additional BO steps.
    LHS init is skipped automatically by run_manager when prior data is seeded,
    so the key decision is n_opt_steps (and optionally discarding the data).
    Updates plan_steps in place then routes to approve_concrete.
    """
    if state.get("status") == "error":
        return {}

    steps = list(state.get("plan_steps") or [])

    ws_indices = [
        i for i, s in enumerate(steps)
        if s.get("tool") == "run_optimization" and s.get("_found")
    ]
    if not ws_indices:
        return {}  # safety: nothing to negotiate

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
        lbl        = " ({})".format(s["label"]) if s.get("label") else ""
        specs      = s.get("param_specs") or []
        fixed_ctx  = s.get("fixed_context") or {}
        fixed_str  = "  ".join("{}={}".format(k, v) for k, v in sorted(fixed_ctx.items()))
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

    # Use the LLM to parse free-form input into per-step patches
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

    # Apply patches to the relevant steps
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
            status_str = "  \u2192 SURROGATE PREDICTION (no simulation or HPC)"
        elif s["tool"] == "run_simulation":
            if found and not isinstance(found, list):
                short_id  = found["id"][:8]
                best_y    = found.get("best_y")
                best_str  = " best=${:.0f}".format(-best_y) if best_y is not None else ""
                status_str = "  \u2192 REUSE [{}]{} \u2014 no new simulation".format(short_id, best_str)
            else:
                status_str = "  \u2192 RUN FRESH"
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
                    "  \u2192 AUTO WARM-START from prior data:\n"
                    + "\n".join("           " + line for line in prior_lines)
                )
            else:
                status_str = "  \u2192 RUN FRESH (no prior optimization data \u2014 LHS warm-up)"
        else:
            status_str = "  \u2192 RUN FRESH"
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
        params     = r.get("params") or {}
        params_str = "  ".join("{}={}".format(k, v) for k, v in sorted(params.items()))
        lines.append("  {} ({}): {}{}".format(
            r.get("label", "?"), params_str, cost_str, best_marker))

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
# 6. Routing
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
    """Route to negotiate_reuse when any opt step has warm-start data, else go straight to approve_concrete."""
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
# 7. Graph assembly
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

_STANDALONE_LOCK = os.path.join(_AGENT_DIR, ".rental_agent.lock")


def _acquire_standalone_lock() -> None:
    """Prevent two standalone rental_agent.py processes from running at once.

    Not called when running under co_scientist (chat_id is provided) because
    each co_scientist session is safe — it uses its own Hero queue tasks
    tracked by task ID, and clear_hero_queue() is no longer called at startup.
    """
    lock_path = _STANDALONE_LOCK
    if os.path.exists(lock_path):
        try:
            pid = int(open(lock_path).read().strip())
            os.kill(pid, 0)   # signal 0 = existence check only
            print(
                "\nERROR: Another rental_agent session is already running (PID {}).\n"
                "  Running two standalone agents at once shares the same Hero queue\n"
                "  and can corrupt each other's results.\n"
                "\n"
                "  Use co_scientist.py to manage multiple parallel investigations:\n"
                "      python co_scientist.py\n"
                "\n"
                "  Or stop the other session first, then retry.".format(pid)
            )
            sys.exit(1)
        except (ProcessLookupError, ValueError, PermissionError):
            pass  # stale lock — process is gone, safe to overwrite

    with open(lock_path, "w") as f:
        f.write(str(os.getpid()))

    import atexit
    atexit.register(lambda: os.path.exists(lock_path) and os.remove(lock_path))


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
    user_request     : The user's goal / research question.
    chat_id          : If set, checkpoint writes are enabled.  Supplied by
                       co_scientist.py when launching a managed session.
    checkpoint_file  : Path to write checkpoint JSON.  Required when chat_id
                       is provided.
    prior_history    : conversation_history from a previous run (for follow-up
                       sessions).
    """
    global _CHAT_ID, _CHECKPOINT_FILE, _CHECKPOINT_STATE

    if not chat_id:
        _acquire_standalone_lock()

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
        _write_checkpoint()  # initial write so the entry exists immediately
        session_name = "co-sci-{}".format(chat_id[:8])
        print("[agent] Session: {}  →  To kill: tmux kill-session -t {}".format(
            session_name, session_name))
        print("[agent] MCP server →  To kill: tmux kill-session -t {}".format(
            _MCP_SESSION_NAME))

    _ensure_server_running()
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
                graph = build_graph(checkpointer=checkpointer)
                config = {"configurable": {"thread_id": chat_id}}
                snap = graph.get_state(config)
                if snap.next:
                    print("[rental_agent] Resuming interrupted session from checkpoint...")
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


def _stop_mcp_server() -> None:
    """Kill the AC MCP server tmux session when this agent script exits."""
    try:
        from adaptive_computing.hpc.local_launcher import stop_manager
        stop_manager(_MCP_SESSION_NAME)
    except Exception:
        pass


if __name__ == "__main__":
    import atexit
    atexit.register(_stop_mcp_server)

    examples = [
        "What storage percentage minimizes daily cost for a facility with 5000 EVs/day "
        "using an Aggressive utility rate?",

        "Compare Moderate vs Aggressive utility rates for a medium-demand facility "
        "(1000 EVs/day, SOC 35). Which is cheaper and by how much?",

        "Survey the full parameter space with LHS sampling and explain which parameters "
        "have the biggest impact on cost.",

        "Conduct a parallel Bayesian optimization (3 initial samples, 1 batch of 5 parallel BO samples) to find the cost-minimizing demand (number of daily EVs) with fixed state of charge = 30, fixed storage=40 percent, and fixed utility rate = Aggressive.",

        "Survey the parameter space holding utility_rate=moderate fixed and the other variables varying. Use the maximum_variance acquisition function and 5 initial samples and 1 batch of 5 parallel BO samples. Then use the surrogate to interpolate to evaluate the point utility_rate=moderate, storage=50, number_of_daily_evs=2000, return_soc=40.",

        "Conduct a parallel Bayesian optimization (4 initial samples + 1 batch of 4 parallel BO samples) to find the cost-minimizing demand and state of charge (2 variable optimization) with fixed storage =20 percent and fixed utility rate=moderate.",
    ]

    parser = argparse.ArgumentParser(
        description="Rental-car electrification co-scientist agent",
        add_help=False,
    )
    parser.add_argument(
        "--chat-id",
        default=None,
        metavar="UUID",
        help="Unique ID for this chat session (set by co_scientist.py).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Path to the checkpoint JSON for this session.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load prior_history from the checkpoint file before starting.",
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default=None,
        help="Research goal.  Pass a single digit to pick an example.",
    )
    # Accept any remaining positional tokens as part of the goal
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Build the request string from positional args
    parts = []
    if args.goal:
        parts.append(args.goal)
    if args.extra:
        parts.extend(args.extra)
    arg_goal = " ".join(parts).strip()

    # Load prior conversation history when resuming
    prior_hist = []
    if args.resume and args.checkpoint:
        try:
            import chat_registry
            ckpt = chat_registry.read_checkpoint(args.checkpoint)
            prior_hist = ckpt.get("conversation_history", [])
            if not arg_goal:
                arg_goal = ckpt.get("user_request", "")
            if prior_hist:
                print("[Resuming with {} prior conversation turn(s)]".format(
                    len(prior_hist)
                ))
        except Exception as exc:
            print("[checkpoint] Warning: could not load prior history: {}".format(exc))

    if arg_goal:
        # Allow passing a single digit to select an example prompt
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
