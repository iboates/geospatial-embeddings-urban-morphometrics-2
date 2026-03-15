"""Global meshedness metric.

Graph-level ratio of the number of circuits to the maximum possible number
of circuits: (E - N + 1) / (2N - 5). A value of 0 indicates a tree network
(no loops); values approaching 1 indicate a dense grid with many loops.

Computed as a single scalar per graph (vehicle and pedestrian), not per node,
so no aggregation is applied. Returns NaN when the graph has too few nodes to
form a valid circuit (fewer than 3 nodes).
"""

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register


def _compute(graph) -> float:
    if graph is None or graph.number_of_nodes() < 3:
        return float("nan")
    return momepy.meshedness(graph, radius=None)


@register("meshedness_global")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Global meshedness (E-N+1)/(2N-5) for vehicle and pedestrian networks."""
    return {
        "meshedness_global_vehicle": _compute(ctx.vehicle_graph),
        "meshedness_global_pedestrian": _compute(ctx.pedestrian_graph),
    }
