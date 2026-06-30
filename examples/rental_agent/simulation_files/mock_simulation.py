#!/usr/bin/env python3
"""
mock_simulation.py — Deterministic mock rental car electrification model.

Reads config.json from the case directory, computes cost analytically,
and writes result.json to the same directory.

Usage (called by job.sh):
    python mock_simulation.py <case_dir>

Config JSON fields:
    utility_rate          str    "Moderate" | "Aggressive"
    storage              float  0 – 100  (storage capacity as a percentage; 0 = grid-only)
    number_of_daily_evs   float  10 – 10000  (continuous)
    return_soc            float  25 – 55  (continuous state-of-charge percentage)

Output (result.json):
    cost   float   Total daily energy cost in USD
"""

import json
import os
import sys
import time


# ---------------------------------------------------------------------------
# Analytic cost model
# ---------------------------------------------------------------------------

# Base electricity cost per EV per day (USD).
# Covers charging infrastructure amortization + energy at grid rate.
_BASE_COST_PER_EV = 50.0

# Aggressive utility rate = peak-demand tariff; raises cost by 50 %.
_UTILITY_MULT = {
    "Moderate":   1.0,
    "Aggressive": 1.5,
}

# Battery storage shifts load away from peak windows, reducing demand charges.
# Discount is piecewise-linearly interpolated over storage percentage 0–100.
_STORAGE_KNOTS_X = [0.0,  25.0, 50.0, 75.0, 100.0]
_STORAGE_KNOTS_Y = [0.00, 0.15, 0.28, 0.38, 0.45]

def _storage_discount(storage_pct: float) -> float:
    """Piecewise-linear interpolation of storage discount for storage_pct in [0, 100]."""
    storage_pct = max(0.0, min(100.0, float(storage_pct)))
    for i in range(len(_STORAGE_KNOTS_X) - 1):
        if storage_pct <= _STORAGE_KNOTS_X[i + 1]:
            t = (storage_pct - _STORAGE_KNOTS_X[i]) / (_STORAGE_KNOTS_X[i + 1] - _STORAGE_KNOTS_X[i])
            return _STORAGE_KNOTS_Y[i] + t * (_STORAGE_KNOTS_Y[i + 1] - _STORAGE_KNOTS_Y[i])
    return _STORAGE_KNOTS_Y[-1]

# Higher return SOC → vehicles arrive more charged → less energy to dispense.
# Discount scales linearly from 0 % at SOC 25 to 25 % at SOC 55.
def _soc_factor(return_soc: float) -> float:
    return_soc = max(25.0, min(55.0, float(return_soc)))
    discount = (return_soc - 25.0) / 30.0 * 0.25
    return 1.0 - discount


def compute_cost(
    utility_rate: str,
    storage: float,
    number_of_daily_evs: float,
    return_soc: float,
) -> float:
    """
    Return total daily cost (USD) for the given configuration.

    Formula:
        cost = BASE_COST_PER_EV
               * number_of_daily_evs
               * utility_mult
               * (1 - storage_discount(storage))
               * soc_factor(return_soc)

    Example outputs (sanity check):
        Moderate,  storage=0,   1 000 EVs, SOC 25 →  $50 000
        Moderate,  storage=100, 1 000 EVs, SOC 55 →  $20 625
        Aggressive,storage=0,   1 000 EVs, SOC 25 →  $75 000
        Aggressive,storage=100,10 000 EVs, SOC 55 →  $206 250
    """
    if utility_rate not in _UTILITY_MULT:
        raise ValueError(f"Unknown utility_rate: {utility_rate!r}")

    cost = (
        _BASE_COST_PER_EV
        * number_of_daily_evs
        * _UTILITY_MULT[utility_rate]
        * (1.0 - _storage_discount(storage))
        * _soc_factor(return_soc)
    )
    return round(cost, 2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <case_dir>")
        sys.exit(1)

    case_dir = sys.argv[1]
    config_path = os.path.join(case_dir, "config.json")
    result_path = os.path.join(case_dir, "result.json")

    if not os.path.exists(config_path):
        print(f"ERROR: config.json not found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    print(f"Config: {json.dumps(config, indent=2)}")

    # Artificial delay so Slurm jobs are visible in squeue during testing.
    #time.sleep(10)

    cost = compute_cost(
        utility_rate=config["utility_rate"],
        storage=float(config["storage"]),
        number_of_daily_evs=float(config["number_of_daily_evs"]),
        return_soc=float(config["return_soc"]),
    )

    result = {"cost": cost}
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Result: cost = ${cost:,.2f}")
    print(f"Wrote result to {result_path}")


if __name__ == "__main__":
    main()
