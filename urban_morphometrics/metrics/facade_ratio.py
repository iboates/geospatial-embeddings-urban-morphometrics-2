"""Facade ratio metric.

Ratio of building area to perimeter (area / perimeter). Computed twice:
- `facade_ratio`: per raw individual building
- `facade_ratio_joined`: per dissolved structure (touching buildings merged)

Higher values indicate more compact structures; lower values indicate elongated
or highly articulated perimeters relative to their area.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics._utils import dissolve_touching


@register("facade_ratio")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Area / perimeter per raw building and per dissolved (joined) structure."""
    b = ctx.buildings_ea
    empty = pd.Series(dtype=float)
    if b.empty:
        return {
            **aggregate_series(empty, "facade_ratio", num_quantiles),
            **aggregate_series(empty, "facade_ratio_joined", num_quantiles),
        }

    result = aggregate_series(momepy.facade_ratio(b), "facade_ratio", num_quantiles)

    dissolved = dissolve_touching(b)
    result.update(aggregate_series(momepy.facade_ratio(dissolved), "facade_ratio_joined", num_quantiles))

    return result
