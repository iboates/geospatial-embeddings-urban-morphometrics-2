"""Mean node degree metric (per node, subgraph radius).

Average node degree within each node's local subgraph of
network_subgraph_radius hops. Captures local network density: high values
indicate many-way intersections in the neighbourhood; low values indicate
mostly dead ends or T-junctions.

Requires node degree to be computed first; done inline before calling momepy.
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
    prefix = f"mean_node_degree_{suffix}"
    empty = aggregate_series(pd.Series(dtype=float), prefix, num_quantiles)
    if graph is None:
        return empty
    momepy.node_degree(graph)
    momepy.mean_node_degree(graph, radius=radius, verbose=False)
    values = focal_nodes_series(graph, "mean_nd", cell_geom)
    return aggregate_series(values, prefix, num_quantiles)


@register("mean_node_degree")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Mean node degree within configurable-radius subgraph for vehicle and pedestrian networks."""
    radius = ctx.config.network_subgraph_radius
    row = {}
    row.update(_compute(ctx.vehicle_graph, "vehicle", ctx._cell_ed, radius, num_quantiles))
    row.update(_compute(ctx.pedestrian_graph, "pedestrian", ctx._cell_ed, radius, num_quantiles))
    return row
