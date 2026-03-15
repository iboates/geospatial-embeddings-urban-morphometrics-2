"""Aggregation helpers for per-feature metric series."""

import numpy as np
import pandas as pd


def aggregate_series(series: pd.Series, prefix: str, num_quantiles: int) -> dict:
    """Summarise a per-feature Series into a flat dict of named statistics.

    Produces mean, median, std, and evenly-spaced quantiles. NaN values are
    dropped before all calculations. If the series is entirely NaN or empty,
    all output values are NaN.

    Args:
        series: Per-feature numeric values (e.g. area per building).
        prefix: Column name prefix (e.g. "floor_area" → "floor_area_mean").
        num_quantiles: Number of quantile bands. For example, 4 produces
            q25, q50, q75, q100 (i.e. quartiles). Labels are "q{percentile}".

    Returns:
        Dict with keys: {prefix}_mean, {prefix}_median, {prefix}_std,
        and {prefix}_q{p} for each quantile percentile.
    """
    clean = series.dropna()

    if clean.empty:
        result = {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
        }
        quantile_probs = np.linspace(0, 1, num_quantiles + 1)[1:]
        for p in quantile_probs:
            label = int(round(p * 100))
            result[f"{prefix}_q{label}"] = np.nan
        return result

    result = {
        f"{prefix}_mean": float(clean.mean()),
        f"{prefix}_median": float(clean.median()),
        f"{prefix}_std": float(clean.std()),
    }

    quantile_probs = np.linspace(0, 1, num_quantiles + 1)[1:]
    quantile_vals = clean.quantile(quantile_probs)
    for p, val in zip(quantile_probs, quantile_vals):
        label = int(round(p * 100))
        result[f"{prefix}_q{label}"] = float(val)

    return result
