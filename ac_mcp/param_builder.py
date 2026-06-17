"""
param_builder.py
================
Converts a JSON-serialisable parameter spec into AdaptiveComputing variable
objects, and builds the Hero task_formatter from those params + fixed_context.

Param spec format (list of dicts):
    {"name": "g_b_val",     "type": "continuous", "min": 0.0, "max": 10.0}
    {"name": "n_layers",    "type": "ordered",    "min": 1,   "max": 8}
    {"name": "neuron_type", "type": "categorical", "categories": ["LIF", "BE-LIF"]}

fixed_context: dict of name → value for parameters that are constant across all
jobs in an experiment (e.g. {"n_e": 100, "neuron_type": "BE-LIF"}).
"""

from __future__ import annotations
from typing import Any


def build_ac_params(param_specs: list[dict]) -> list:
    """Convert JSON param specs to a list of AC variable objects."""
    from adaptive_computing.datasets import (
        ContinuousVariable,
        OrderedVariable,
        CategoricalVariable,
    )

    ac_vars = []
    for spec in param_specs:
        ptype = spec["type"].lower()
        if ptype == "continuous":
            ac_vars.append(ContinuousVariable(min=float(spec["min"]),
                                              max=float(spec["max"])))
        elif ptype == "ordered":
            ac_vars.append(OrderedVariable(min_val=int(spec["min"]),
                                           max_val=int(spec["max"])))
        elif ptype == "categorical":
            ac_vars.append(CategoricalVariable(spec["categories"]))
        else:
            raise ValueError(f"Unknown param type: {ptype!r}")
    return ac_vars


class _TaskFormatter:
    """Picklable callable that maps x_data_i floats → Hero task metadata dict."""
    def __init__(self, param_specs: list[dict], fixed_context: dict,
                 machine_names: list[str]):
        self.param_specs    = param_specs
        self.fixed_context  = fixed_context
        self.machine_names  = machine_names

    def __call__(self, x_data_i: Any) -> dict:
        meta: dict = {"y_data": None}
        for i, spec in enumerate(self.param_specs):
            raw   = float(x_data_i[i])
            ptype = spec["type"].lower()
            if ptype == "categorical":
                meta[spec["name"]] = spec["categories"][int(round(raw))]
            elif ptype == "ordered":
                meta[spec["name"]] = int(round(raw))
            else:
                meta[spec["name"]] = raw
        meta.update(self.fixed_context)
        meta["slurm_job_id"] = {m: -1    for m in self.machine_names}
        meta["running"]      = {m: False for m in self.machine_names}
        return meta


class _EvalFormatter:
    """Picklable callable for evaluation-only runs; maps x_data_i index → job metadata."""
    def __init__(self, jobs: list[dict], machine_names: list[str]):
        self.jobs          = jobs
        self.machine_names = machine_names

    def __call__(self, x_data_i: Any) -> dict:
        idx  = int(round(float(x_data_i[0])))
        meta = dict(self.jobs[idx])
        meta.setdefault("y_data", None)
        meta["slurm_job_id"] = {m: -1    for m in self.machine_names}
        meta["running"]      = {m: False for m in self.machine_names}
        return meta


def build_task_formatter(param_specs: list[dict], fixed_context: dict,
                         machine_names: list[str]) -> _TaskFormatter:
    """
    Return a picklable Hero task_formatter.

    The formatter maps x_data_i (1-D float array, one value per param_spec)
    to a metadata dict containing decoded param values, fixed_context fields,
    and Hero bookkeeping keys (slurm_job_id, running).
    """
    return _TaskFormatter(param_specs, fixed_context, machine_names)


def build_evaluation_formatter(jobs: list[dict],
                               machine_names: list[str]) -> _EvalFormatter:
    """
    Formatter for evaluation-only (no surrogate) runs.

    x_data encodes a job index [0, 1, 2, ...]; the formatter looks up the full
    metadata from the pre-built `jobs` list.
    """
    return _EvalFormatter(jobs, machine_names)
