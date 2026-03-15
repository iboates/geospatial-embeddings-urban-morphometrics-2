"""Rectangularity metric.

Ratio of building area to its minimum rotated bounding rectangle area.
Values near 1 indicate rectangular footprints.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("rectangularity")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Ratio of building area to its minimum rotated bounding rectangle (0–1)."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "rectangularity", num_quantiles)
    return aggregate_series(momepy.rectangularity(b), "rectangularity", num_quantiles)
