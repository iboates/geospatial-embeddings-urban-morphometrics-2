"""Metric registry and computation entry point.

Each metric is a callable registered in REGISTRY under its name string.
A metric callable has the signature:

    def compute(ctx: CellContext, num_quantiles: int) -> dict[str, float]:
        ...

It receives the CellContext for the current cell and returns a flat dict of
{column_name: value} pairs ready to become columns in the results GeoDataFrame.
"""

import logging
from typing import Callable

from urban_morphometrics.cell_context import CellContext

log = logging.getLogger(__name__)

# Maps metric name → compute function.
# Populated by individual metric modules via register().
REGISTRY: dict[str, Callable] = {}


def register(name: str):
    """Decorator to register a metric compute function under the given name."""
    def decorator(fn: Callable) -> Callable:
        REGISTRY[name] = fn
        return fn
    return decorator


def compute_metrics(
    ctx: CellContext,
    metric_names: list[str],
    num_quantiles: int,
) -> dict:
    """Compute all requested metrics for a single cell.

    Args:
        ctx: CellContext for the cell being processed.
        metric_names: List of metric names to compute, or ["all"] for every
            registered metric.
        num_quantiles: Passed through to each metric's aggregation step.

    Returns:
        Flat dict of {column_name: value} for all computed metrics combined.
    """
    if metric_names == ["all"]:
        names = list(REGISTRY.keys())
    else:
        unknown = [n for n in metric_names if n not in REGISTRY]
        if unknown:
            raise ValueError(f"Unknown metrics: {', '.join(unknown)}. Available: {', '.join(REGISTRY)}")
        names = metric_names

    row = {}
    for name in names:
        try:
            result = REGISTRY[name](ctx, num_quantiles)
            row.update(result)
        except Exception:
            log.warning("Metric '%s' failed for region %s", name, ctx.region_id, exc_info=True)
    return row
