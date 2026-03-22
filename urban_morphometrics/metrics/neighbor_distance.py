"""Neighbour distance metric.

Mean distance from each building to its Delaunay-triangulation neighbours.
Captures typical spacing between nearby buildings. Low values → dense urban
fabric; high values → sparse suburban or rural settlement patterns.

Requires neighbourhood context so Delaunay triangulation is not artificially
clipped at the cell boundary.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("neighbor_distance")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Mean distance from each building to its Delaunay-triangulation neighbours (m)."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "neighbor_distance", num_quantiles)

    delaunay = ctx.delaunay_graph
    if delaunay is None:
        return aggregate_series(pd.Series(dtype=float), "neighbor_distance", num_quantiles)

    all_b = ctx.focal_plus_neighbourhood_buildings
    values = momepy.neighbor_distance(all_b, delaunay)
    focal_values = values.reindex(b.index)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(neighbor_distance=focal_values), features_dir / "neighbor_distance.gpkg")
    return aggregate_series(focal_values, "neighbor_distance", num_quantiles)
