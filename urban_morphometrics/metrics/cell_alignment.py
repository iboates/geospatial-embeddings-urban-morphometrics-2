"""Cell alignment metric.

Deviation between each building's orientation and the orientation of its
morphological tessellation cell. Low values mean buildings are aligned with
their local Voronoi partition (typical of grid-planned areas); high values
indicate buildings rotated relative to the urban fabric around them.

Requires neighbourhood context so tessellation cells at the cell boundary are
correctly computed.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("cell_alignment")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Deviation between building orientation and its tessellation cell orientation (°)."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "cell_alignment", num_quantiles)

    tess = ctx.tessellation
    if tess is None or tess.empty:
        return aggregate_series(pd.Series(dtype=float), "cell_alignment", num_quantiles)

    all_b = ctx.focal_plus_neighbourhood_buildings
    building_orient = momepy.orientation(all_b)
    tess_orient = momepy.orientation(tess)

    # cell_alignment does (left - right).abs(); pandas aligns by index automatically,
    # so missing tessellation entries produce NaN and are excluded from aggregation.
    values = momepy.cell_alignment(building_orient, tess_orient)
    return aggregate_series(values.reindex(b.index), "cell_alignment", num_quantiles)
