"""Building adjacency metric.

Ratio of the number of wall-sharing (rook contiguous) neighbours to the number
of KNN neighbours for each building. High values indicate predominantly
attached/terraced buildings; zero means a free-standing building with no
touching neighbours within its KNN set.

Requires neighbourhood context so both the rook contiguity and the KNN graph
extend beyond the focal cell boundary.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("building_adjacency")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Ratio of wall-sharing neighbours to KNN neighbours per focal building."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "building_adjacency", num_quantiles)

    contiguity = ctx.contiguity_graph
    knn = ctx.knn_graph
    if contiguity is None or knn is None:
        return aggregate_series(pd.Series(dtype=float), "building_adjacency", num_quantiles)

    values = momepy.building_adjacency(contiguity, knn)
    return aggregate_series(values.reindex(b.index), "building_adjacency", num_quantiles)
