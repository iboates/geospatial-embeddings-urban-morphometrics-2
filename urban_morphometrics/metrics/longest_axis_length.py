"""Longest axis length metric.

Wraps momepy.longest_axis_length: the diameter of the minimum bounding circle
of each building footprint. Measures the longest dimension of the building.
Uses equal-area CRS.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("longest_axis_length")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Compute longest axis length statistics for the focal cell.

    The longest axis is the diameter of the minimum bounding circle of each
    building footprint. Values are in metres (equal-area CRS).
    """
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series([], dtype=float), "longest_axis_length", num_quantiles)

    values = momepy.longest_axis_length(b)
    return aggregate_series(values, "longest_axis_length", num_quantiles)
