"""Courtyards metric.

Number of courtyards within the joined structure each building belongs to.
A courtyard is an enclosed open space surrounded by connected buildings.

Requires neighbourhood context so buildings near the cell boundary are connected
to their neighbours across the boundary, avoiding false courtyard splits at the
cell edge.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("courtyards")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Number of courtyards per building's joined structure."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "courtyards", num_quantiles)

    all_b = ctx.focal_plus_neighbourhood_buildings
    graph = ctx.contiguity_graph
    if graph is None:
        return aggregate_series(pd.Series(dtype=float), "courtyards", num_quantiles)

    values = momepy.courtyards(all_b, graph)
    return aggregate_series(values.reindex(b.index), "courtyards", num_quantiles)
