"""Node clustering metric.

Number of triangles in the street network that share each node, normalised by
the number of edges at that node. High values indicate dense, well-connected
local networks; zero values indicate tree-like (acyclic) patterns.

Computed for both vehicle and pedestrian networks. Focal nodes only.
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.street_graph import focal_nodes_series


def _compute(graph, suffix, cell_geom, num_quantiles) -> dict:
    prefix = f"clustering_{suffix}"
    empty = aggregate_series(pd.Series(dtype=float), prefix, num_quantiles)
    if graph is None:
        return empty
    graph = momepy.clustering(graph)
    values = focal_nodes_series(graph, "cluster", cell_geom)
    return aggregate_series(values, prefix, num_quantiles)


@register("clustering")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Node clustering coefficient distribution for vehicle and pedestrian networks."""
    row = {}
    row.update(_compute(ctx.vehicle_graph, "vehicle", ctx._cell_ed, num_quantiles))
    row.update(_compute(ctx.pedestrian_graph, "pedestrian", ctx._cell_ed, num_quantiles))
    return row
