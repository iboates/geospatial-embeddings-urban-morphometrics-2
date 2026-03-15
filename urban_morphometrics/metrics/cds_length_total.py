"""Total cul-de-sac (CDS) length metric.

Total length (metres) of cul-de-sac street segments in the network. A
cul-de-sac segment connects a dead-end node (degree 1) to its junction.
Higher values indicate more suburban, tree-like street patterns; lower
values (or zero) indicate well-connected grid-like patterns.

Computed as a single scalar per graph (vehicle and pedestrian). Requires
node degree to be set; this is computed inline before calling momepy.
"""

import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register


def _compute(graph) -> float:
    if graph is None or graph.number_of_nodes() < 2:
        return float("nan")
    graph = momepy.node_degree(graph)
    return float(momepy.cds_length(graph, radius=None))


@register("cds_length_total")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Total cul-de-sac segment length (m) for vehicle and pedestrian networks."""
    return {
        "cds_length_total_vehicle": _compute(ctx.vehicle_graph),
        "cds_length_total_pedestrian": _compute(ctx.pedestrian_graph),
    }
