"""Perimeter wall metric.

Wraps momepy.perimeter_wall: buildings that share walls are dissolved into
unified structures. The returned perimeter is the outer boundary length of
each merged structure, not the sum of individual building perimeters.
Uses equal-area CRS. Values are in metres.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("perimeter_wall")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Compute perimeter wall statistics for the focal cell.

    Touching buildings are dissolved and the outer perimeter of each merged
    structure is measured. High values indicate large building blocks or
    terraces; low values indicate isolated small buildings. Values in metres.
    """
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series([], dtype=float), "perimeter_wall", num_quantiles)

    values = momepy.perimeter_wall(b)
    return aggregate_series(values, "perimeter_wall", num_quantiles)
