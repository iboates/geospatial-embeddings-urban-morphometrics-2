"""Betweenness centrality metric.

Number of shortest paths (weighted by edge length) between all node pairs
that pass through each node, normalised by the total number of pairs.
High values identify nodes that act as bridges or bottlenecks in the network.

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
    prefix = f"betweenness_centrality_{suffix}"
    empty = aggregate_series(pd.Series(dtype=float), prefix, num_quantiles)
    if graph is None:
        return empty
    graph = momepy.betweenness_centrality(graph, verbose=False)
    values = focal_nodes_series(graph, "betweenness", cell_geom)
    return aggregate_series(values, prefix, num_quantiles)


@register("betweenness_centrality")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Betweenness centrality distribution for vehicle and pedestrian networks."""
    row = {}
    row.update(_compute(ctx.vehicle_graph, "vehicle", ctx._cell_ed, num_quantiles))
    row.update(_compute(ctx.pedestrian_graph, "pedestrian", ctx._cell_ed, num_quantiles))
    return row
