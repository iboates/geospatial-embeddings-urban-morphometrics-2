"""Local edge-node ratio metric (per node, subgraph radius).

For each node, edge-node ratio is computed within its local subgraph of
network_subgraph_radius hops: E_local / N_local. Values above 1 indicate a
network with loops (cyclic); values at or below 1 indicate tree-like structure.
This is the per-node variant; the global variant is in step 11.

The subgraph radius is controlled by MetricConfig.network_subgraph_radius
(default: 5). Computed for both vehicle and pedestrian networks. Focal nodes only.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.street_graph import focal_nodes_series


def _compute(graph, suffix, cell_geom, radius, num_quantiles) -> dict:
    prefix = f"edge_node_ratio_{suffix}"
    empty = aggregate_series(pd.Series(dtype=float), prefix, num_quantiles)
    if graph is None:
        return empty
    momepy.edge_node_ratio(graph, radius=radius, verbose=False)
    values = focal_nodes_series(graph, "edge_node_ratio", cell_geom)
    return aggregate_series(values, prefix, num_quantiles)


@register("edge_node_ratio")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Local edge-node ratio (configurable-radius subgraph) for vehicle and pedestrian networks."""
    radius = ctx.config.network_subgraph_radius
    row = {}
    row.update(_compute(ctx.vehicle_graph, "vehicle", ctx._cell_ed, radius, num_quantiles))
    row.update(_compute(ctx.pedestrian_graph, "pedestrian", ctx._cell_ed, radius, num_quantiles))
    return row
