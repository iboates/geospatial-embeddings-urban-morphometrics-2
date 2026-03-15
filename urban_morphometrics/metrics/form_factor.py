"""Form factor metric.

surface / volume^(2/3), where surface = (perimeter × height) + area.
A 3D compactness measure using building height. Low values indicate compact
blocks; high values indicate thin or tall buildings.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("form_factor")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """surface / volume^(2/3) per building using resolved OSM heights."""
    b = ctx.buildings_with_height
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "form_factor", num_quantiles)
    return aggregate_series(momepy.form_factor(b, b["height"]), "form_factor", num_quantiles)
