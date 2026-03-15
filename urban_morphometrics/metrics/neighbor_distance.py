"""Neighbour distance metric.

Mean distance from each building to its Delaunay-triangulation neighbours.
Captures typical spacing between nearby buildings. Low values → dense urban
fabric; high values → sparse suburban or rural settlement patterns.

Requires neighbourhood context so Delaunay triangulation is not artificially
clipped at the cell boundary.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("neighbor_distance")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Mean distance from each building to its Delaunay-triangulation neighbours (m)."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "neighbor_distance", num_quantiles)

    delaunay = ctx.delaunay_graph
    if delaunay is None:
        return aggregate_series(pd.Series(dtype=float), "neighbor_distance", num_quantiles)

    all_b = ctx.focal_plus_neighbourhood_buildings
    values = momepy.neighbor_distance(all_b, delaunay)
    return aggregate_series(values.reindex(b.index), "neighbor_distance", num_quantiles)
