"""Mean node distance metric.

Mean length of edges connected to each node (metres). Captures the typical
block length or street segment length around each intersection. Short mean
distances indicate fine-grained networks; long distances indicate sparse or
arterial networks.

Computed for both vehicle and pedestrian networks. Focal nodes only.
"""

from pathlib import Path
import warnings

import pandas as pd
import momepy

# momepy.mean_node_dist computes np.mean over successor edge lengths. Nodes at
# the boundary of a one-way street with no outgoing edges produce an empty list,
# triggering these numpy warnings. This is expected for clipped subgraphs.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.street_graph import focal_nodes_series
from urban_morphometrics.metrics.features import write_features


def _compute(graph, suffix, cell_geom, num_quantiles, features_dir=None) -> dict:
    prefix = f"mean_node_dist_{suffix}"
    empty = aggregate_series(pd.Series(dtype=float), prefix, num_quantiles)
    if graph is None:
        return empty
    graph = momepy.mean_node_dist(graph, verbose=False)
    values = focal_nodes_series(graph, "meanlen", cell_geom)
    if features_dir is not None and not values.empty:
        nodes_gdf, _ = momepy.nx_to_gdf(graph)
        export_gdf = nodes_gdf[["geometry"]].copy()
        export_gdf[prefix] = values
        write_features(export_gdf.dropna(subset=[prefix]), features_dir / f"{prefix}.gpkg")
    return aggregate_series(values, prefix, num_quantiles)


@register("mean_node_dist")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Mean edge length per node (m) for vehicle and pedestrian networks."""
    row = {}
    row.update(_compute(ctx.vehicle_graph, "vehicle", ctx._cell_ed, num_quantiles, features_dir))
    row.update(_compute(ctx.pedestrian_graph, "pedestrian", ctx._cell_ed, num_quantiles, features_dir))
    return row
