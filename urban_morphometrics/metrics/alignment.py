"""Alignment metric.

Consistency of orientation among neighbouring buildings, computed via a KNN
spatial graph (default k=15, configurable via MetricConfig.knn_k). Low values
indicate a uniform grid; high values indicate heterogeneous orientations.

Requires neighbourhood context so KNN neighbours of edge buildings are not
artificially restricted to buildings inside the focal cell.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("alignment")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Mean orientation deviation between each building and its KNN neighbours."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "alignment", num_quantiles)

    knn = ctx.knn_graph
    if knn is None:
        return aggregate_series(pd.Series(dtype=float), "alignment", num_quantiles)

    all_b = ctx.focal_plus_neighbourhood_buildings
    orient = momepy.orientation(all_b)
    values = momepy.alignment(orient, knn)
    focal_values = values.reindex(b.index)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(alignment=focal_values), features_dir / "alignment.gpkg")
    return aggregate_series(focal_values, "alignment", num_quantiles)
