"""Global gamma metric.

Graph-level ratio of observed edges to the maximum possible edges:
E / (3(N - 2)). Ranges from 0 (no connectivity) to 1 (maximally connected
planar graph). Complements meshedness: gamma captures edge density while
meshedness captures circuit density.

Computed as a single scalar per graph (vehicle and pedestrian). Returns NaN
when the graph has fewer than 3 nodes.
"""

import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register


def _compute(graph) -> float:
    if graph is None or graph.number_of_nodes() < 3:
        return float("nan")
    return momepy.gamma(graph, radius=None)


@register("gamma_global")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Global gamma E/(3(N-2)) for vehicle and pedestrian networks."""
    return {
        "gamma_global_vehicle": _compute(ctx.vehicle_graph),
        "gamma_global_pedestrian": _compute(ctx.pedestrian_graph),
    }
