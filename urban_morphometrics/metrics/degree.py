"""Node degree metric.

Number of street segments connected to each intersection node. Degree 1 = dead
end, degree 3 = T-junction, degree 4 = crossroads. The distribution of node
degrees characterises the connectivity texture of the street network.

Computed for both vehicle (directed) and pedestrian (undirected) networks.
Focal nodes only (those whose position lies within the study cell).
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
    prefix = f"degree_{suffix}"
    empty = aggregate_series(pd.Series(dtype=float), prefix, num_quantiles)
    if graph is None:
        return empty
    graph = momepy.node_degree(graph)
    values = focal_nodes_series(graph, "degree", cell_geom)
    if features_dir is not None and not values.empty:
        nodes_gdf, _ = momepy.nx_to_gdf(graph)
        export_gdf = nodes_gdf[["geometry"]].copy()
        export_gdf[prefix] = values
        write_features(export_gdf.dropna(subset=[prefix]), features_dir / f"{prefix}.gpkg")
    return aggregate_series(values, prefix, num_quantiles)


@register("degree")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Node degree distribution for vehicle and pedestrian street networks."""
    row = {}
    row.update(_compute(ctx.vehicle_graph, "vehicle", ctx._cell_ed, num_quantiles, features_dir))
    row.update(_compute(ctx.pedestrian_graph, "pedestrian", ctx._cell_ed, num_quantiles, features_dir))
    return row
