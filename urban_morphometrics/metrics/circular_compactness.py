"""Circular compactness metric.

Ratio of building area to the area of its minimum bounding circle.
Values near 1 indicate circular footprints; lower values indicate irregular shapes.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("circular_compactness")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Ratio of building area to its minimum bounding circle area (0–1)."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "circular_compactness", num_quantiles)
    return aggregate_series(momepy.circular_compactness(b), "circular_compactness", num_quantiles)
