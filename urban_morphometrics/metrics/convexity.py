"""Convexity metric.

Ratio of building area to its convex hull area. Values near 1 mean the footprint
has no concavities; lower values indicate L-shapes, courtyards, or concave forms.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("convexity")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Ratio of building area to its convex hull area (0–1)."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "convexity", num_quantiles)
    return aggregate_series(momepy.convexity(b), "convexity", num_quantiles)
