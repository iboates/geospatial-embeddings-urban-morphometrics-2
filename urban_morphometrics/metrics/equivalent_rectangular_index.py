"""Equivalent rectangular index (ERI) metric.

Ratio comparing the building's area and perimeter to an equivalent rectangle.
High ERI → rectangular footprints (row houses, slabs); low ERI → complex
footprints (L-shapes, courtyards).
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("equivalent_rectangular_index")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """ERI per building — compares area and perimeter to an equivalent rectangle."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "equivalent_rectangular_index", num_quantiles)
    return aggregate_series(momepy.equivalent_rectangular_index(b), "equivalent_rectangular_index", num_quantiles)
