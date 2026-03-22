"""Local cyclomatic complexity metric (per node, subgraph radius).

For each node, cyclomatic complexity within its local subgraph of
network_subgraph_radius hops: E_local - N_local + 1. Counts the number of
independent circuits (loops) in the local neighbourhood. Zero means acyclic
(tree); positive values indicate grid-like redundancy. This is the per-node variant.

The subgraph radius is controlled by MetricConfig.network_subgraph_radius
(default: 5). Computed for both vehicle and pedestrian networks. Focal nodes only.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.street_graph import focal_nodes_series
from urban_morphometrics.metrics.features import write_features


def _compute(graph, suffix, cell_geom, radius, num_quantiles, features_dir=None) -> dict:
    prefix = f"cyclomatic_{suffix}"
    empty = aggregate_series(pd.Series(dtype=float), prefix, num_quantiles)
    if graph is None:
        return empty
    graph = momepy.cyclomatic(graph, radius=radius, verbose=False)
    values = focal_nodes_series(graph, "cyclomatic", cell_geom)
    if features_dir is not None and not values.empty:
        nodes_gdf, _ = momepy.nx_to_gdf(graph)
        export_gdf = nodes_gdf[["geometry"]].copy()
        export_gdf[prefix] = values
        write_features(export_gdf.dropna(subset=[prefix]), features_dir / f"{prefix}.gpkg")
    return aggregate_series(values, prefix, num_quantiles)


@register("cyclomatic")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Local cyclomatic complexity (configurable-radius subgraph) for vehicle and pedestrian networks."""
    radius = ctx.config.network_subgraph_radius
    row = {}
    row.update(_compute(ctx.vehicle_graph, "vehicle", ctx._cell_ed, radius, num_quantiles, features_dir))
    row.update(_compute(ctx.pedestrian_graph, "pedestrian", ctx._cell_ed, radius, num_quantiles, features_dir))
    return row
