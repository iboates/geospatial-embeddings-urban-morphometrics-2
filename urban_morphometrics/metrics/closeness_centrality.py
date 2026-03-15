"""Closeness centrality metric.

Inverse of the mean shortest-path distance from each node to all other
reachable nodes, weighted by edge length ('mm_len'). High closeness means a
node is close to all others on average — well-positioned in the network.

Computed for both vehicle (directed) and pedestrian (undirected) networks.
Focal nodes only.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.street_graph import focal_nodes_series


def _compute(graph, suffix, cell_geom, num_quantiles) -> dict:
    prefix = f"closeness_centrality_{suffix}"
    empty = aggregate_series(pd.Series(dtype=float), prefix, num_quantiles)
    if graph is None:
        return empty
    momepy.closeness_centrality(graph, verbose=False)
    values = focal_nodes_series(graph, "closeness", cell_geom)
    return aggregate_series(values, prefix, num_quantiles)


@register("closeness_centrality")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Closeness centrality distribution for vehicle and pedestrian networks."""
    row = {}
    row.update(_compute(ctx.vehicle_graph, "vehicle", ctx._cell_ed, num_quantiles))
    row.update(_compute(ctx.pedestrian_graph, "pedestrian", ctx._cell_ed, num_quantiles))
    return row
