"""Square compactness metric.

(4 * sqrt(area) / perimeter)^2. Measures how efficiently the perimeter encloses
area. A perfect square scores 1; lower values indicate less compact shapes.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("square_compactness")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """(4√area / perimeter)² per building — 1 for a perfect square."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "square_compactness", num_quantiles)
    return aggregate_series(momepy.square_compactness(b), "square_compactness", num_quantiles)
