"""Straightness centrality metric.

Ratio of Euclidean distance to network distance for shortest paths from each
node to all other reachable nodes (normalised). Values near 1 indicate that
the network routes are almost as direct as the straight-line path (grid-like);
lower values indicate detours (organic or discontinuous networks).

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
    prefix = f"straightness_centrality_{suffix}"
    empty = aggregate_series(pd.Series(dtype=float), prefix, num_quantiles)
    if graph is None:
        return empty
    momepy.straightness_centrality(graph, verbose=False)
    values = focal_nodes_series(graph, "straightness", cell_geom)
    return aggregate_series(values, prefix, num_quantiles)


@register("straightness_centrality")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Straightness centrality distribution for vehicle and pedestrian networks."""
    row = {}
    row.update(_compute(ctx.vehicle_graph, "vehicle", ctx._cell_ed, num_quantiles))
    row.update(_compute(ctx.pedestrian_graph, "pedestrian", ctx._cell_ed, num_quantiles))
    return row
