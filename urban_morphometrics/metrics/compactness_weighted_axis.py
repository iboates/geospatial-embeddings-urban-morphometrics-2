"""Compactness-weighted axis metric.

d × (4/π − 16·area / perimeter²), where d is the longest axis length.
Combines longest axis length with compactness to measure how efficiently a
polygon fills space relative to its principal axes. Computed twice:
- `compactness_weighted_axis`: per raw individual building
- `compactness_weighted_axis_joined`: per dissolved structure (touching buildings merged)
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics._utils import dissolve_touching


@register("compactness_weighted_axis")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Longest-axis-weighted compactness per raw building and per dissolved structure."""
    b = ctx.buildings_ea
    empty = pd.Series(dtype=float)
    if b.empty:
        return {
            **aggregate_series(empty, "compactness_weighted_axis", num_quantiles),
            **aggregate_series(empty, "compactness_weighted_axis_joined", num_quantiles),
        }

    result = aggregate_series(momepy.compactness_weighted_axis(b), "compactness_weighted_axis", num_quantiles)

    dissolved = dissolve_touching(b)
    result.update(aggregate_series(momepy.compactness_weighted_axis(dissolved), "compactness_weighted_axis_joined", num_quantiles))

    return result
