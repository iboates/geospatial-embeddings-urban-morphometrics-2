"""Orientation metric.

Deviation of each building's longest axis from cardinal directions (0°–45° range).
Low mean orientation → grid-aligned urban fabric; high values → organic layout.
No neighbourhood context needed — computed per individual building footprint.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("orientation")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Deviation of each building's longest axis from cardinal directions (0°–45°)."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "orientation", num_quantiles)
    return aggregate_series(momepy.orientation(b), "orientation", num_quantiles)
