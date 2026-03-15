"""Elongation metric.

Ratio of the shorter to the longer side of the minimum bounding rectangle.
Values near 1 indicate compact/square footprints; low values indicate elongated
buildings such as row houses or linear strips.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("elongation")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Shorter/longer side ratio of the minimum bounding rectangle per building (0–1)."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "elongation", num_quantiles)
    return aggregate_series(momepy.elongation(b), "elongation", num_quantiles)
