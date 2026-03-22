"""Straightness centrality metric.

Ratio of Euclidean distance to network distance for shortest paths from each
node to all other reachable nodes (normalised). Values near 1 indicate that
the network routes are almost as direct as the straight-line path (grid-like);
lower values indicate detours (organic or discontinuous networks).

Computed for both vehicle (directed) and pedestrian (undirected) networks.
Focal nodes only.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.street_graph import focal_nodes_series
from urban_morphometrics.metrics.features import write_features


def _compute(graph, suffix, cell_geom, num_quantiles, features_dir=None) -> dict:
    prefix = f"straightness_centrality_{suffix}"
    empty = aggregate_series(pd.Series(dtype=float), prefix, num_quantiles)
    if graph is None:
        return empty
    graph = momepy.straightness_centrality(graph, verbose=False)
    values = focal_nodes_series(graph, "straightness", cell_geom)
    if features_dir is not None and not values.empty:
        nodes_gdf, _ = momepy.nx_to_gdf(graph)
        export_gdf = nodes_gdf[["geometry"]].copy()
        export_gdf[prefix] = values
        write_features(export_gdf.dropna(subset=[prefix]), features_dir / f"{prefix}.gpkg")
    return aggregate_series(values, prefix, num_quantiles)


@register("straightness_centrality")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Straightness centrality distribution for vehicle and pedestrian networks."""
    row = {}
    row.update(_compute(ctx.vehicle_graph, "vehicle", ctx._cell_ed, num_quantiles, features_dir))
    row.update(_compute(ctx.pedestrian_graph, "pedestrian", ctx._cell_ed, num_quantiles, features_dir))
    return row
