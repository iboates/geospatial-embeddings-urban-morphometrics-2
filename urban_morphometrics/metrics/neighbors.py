"""Neighbours metric.

Number of neighbours for each building's morphological tessellation cell,
measured by queen contiguity on the tessellation. Captures how many buildings
surround each building in the local urban fabric.

Requires neighbourhood context so the tessellation is not clipped at the cell
boundary, which would produce artificially small tessellation cells for buildings
near the edge.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("neighbors")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Number of queen-contiguous tessellation neighbours per focal building."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "neighbors", num_quantiles)

    tess = ctx.tessellation
    queen = ctx.tessellation_queen_graph
    if tess is None or tess.empty or queen is None:
        return aggregate_series(pd.Series(dtype=float), "neighbors", num_quantiles)

    values = momepy.neighbors(tess, queen)
    return aggregate_series(values.reindex(b.index), "neighbors", num_quantiles)
