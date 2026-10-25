"""Forecast errors and an explicitly hypothetical rental-capacity exercise."""

import numpy as np
import pandas as pd


def paired_values(actual, predicted):
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError("Expected nonempty, matching one-dimensional arrays.")
    if (
        not np.isfinite(y).all()
        or not np.isfinite(p).all()
        or (y < 0).any()
        or (p < 0).any()
    ):
        raise ValueError("Actuals and predictions must be finite and nonnegative.")
    return y, p


def forecast_metrics(actual, predicted):
    """WAPE and bias are percentages; undefined when actual totals are zero."""
    y, p = paired_values(actual, predicted)
    error = p - y
    total = y.sum()
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape_pct": float(100 * np.abs(error).sum() / total) if total else None,
        "bias_pct": float(100 * error.sum() / total) if total else None,
    }


def capacity_scenarios(actual, predicted, excess_cost=0.5, shortage_cost=3.0):
    """Compare hourly rental slots, not physical fleet size."""
    y, p = paired_values(actual, predicted)
    if (
        not np.isfinite([excess_cost, shortage_cost]).all()
        or min(excess_cost, shortage_cost) < 0
    ):
        raise ValueError("Hypothetical costs must be finite and nonnegative.")
    rows = []
    for buffer in [0.05, 0.15, 0.30]:
        slots = np.ceil(p * (1 + buffer))
        shortage = np.maximum(y - slots, 0).sum()
        excess = np.maximum(slots - y, 0).sum()
        rows.append(
            {
                "buffer_pct": int(round(buffer * 100)),
                "shortage_slots": float(shortage),
                "excess_slots": float(excess),
                "shortage_hours_pct": float(100 * (y > slots).mean()),
                "covered_rentals_pct": float(100 * (1 - shortage / y.sum()))
                if y.sum()
                else None,
                "hypothetical_cost": float(
                    excess * excess_cost + shortage * shortage_cost
                ),
            }
        )
    return pd.DataFrame(rows)
