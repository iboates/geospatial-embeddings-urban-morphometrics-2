"""Mean node distance metric.

Mean length of edges connected to each node (metres). Captures the typical
block length or street segment length around each intersection. Short mean
distances indicate fine-grained networks; long distances indicate sparse or
arterial networks.

Computed for both vehicle and pedestrian networks. Focal nodes only.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.street_graph import focal_nodes_series


def _compute(graph, suffix, cell_geom, num_quantiles) -> dict:
    prefix = f"mean_node_dist_{suffix}"
    empty = aggregate_series(pd.Series(dtype=float), prefix, num_quantiles)
    if graph is None:
        return empty
    graph = momepy.mean_node_dist(graph)
    values = focal_nodes_series(graph, "meanlen", cell_geom)
    return aggregate_series(values, prefix, num_quantiles)


@register("mean_node_dist")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Mean edge length per node (m) for vehicle and pedestrian networks."""
    row = {}
    row.update(_compute(ctx.vehicle_graph, "vehicle", ctx._cell_ed, num_quantiles))
    row.update(_compute(ctx.pedestrian_graph, "pedestrian", ctx._cell_ed, num_quantiles))
    return row
