"""Node proportion metrics.

Proportion of street network nodes with three-way, four-way, and dead-end
(degree-1) connections. These proportions characterise the connectivity
structure of the network:

  - proportion_three:  share of T- or Y-intersections (degree 3)
  - proportion_four:   share of cross-intersections (degree 4)
  - proportion_dead:   share of dead ends / cul-de-sac termini (degree 1)

All three sum to ≤ 1; the remainder are nodes with other degrees (degree 0,
2, or ≥ 5). Computed as global graph-level scalars (one value per network
type per cell), not per node.

Requires node degree to be set; this is computed inline before calling momepy.
"""

import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register


def _compute(graph, suffix: str) -> dict:
    if graph is None or graph.number_of_nodes() < 2:
        return {
            f"proportion_three_{suffix}": float("nan"),
            f"proportion_four_{suffix}": float("nan"),
            f"proportion_dead_{suffix}": float("nan"),
        }
    momepy.node_degree(graph)
    result = momepy.proportion(
        graph,
        radius=None,
        three=f"proportion_three_{suffix}",
        four=f"proportion_four_{suffix}",
        dead=f"proportion_dead_{suffix}",
    )
    return result


@register("proportion")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Proportion of 3-way, 4-way, and dead-end nodes for vehicle and pedestrian networks."""
    row = {}
    row.update(_compute(ctx.vehicle_graph, "vehicle"))
    row.update(_compute(ctx.pedestrian_graph, "pedestrian"))
    return row
