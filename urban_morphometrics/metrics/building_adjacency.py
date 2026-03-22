"""Building adjacency metric.

Ratio of the number of wall-sharing (rook contiguous) neighbours to the number
of KNN neighbours for each building. High values indicate predominantly
attached/terraced buildings; zero means a free-standing building with no
touching neighbours within its KNN set.

Requires neighbourhood context so both the rook contiguity and the KNN graph
extend beyond the focal cell boundary.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("building_adjacency")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Ratio of wall-sharing neighbours to KNN neighbours per focal building."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "building_adjacency", num_quantiles)

    contiguity = ctx.contiguity_graph
    knn = ctx.knn_graph
    if contiguity is None or knn is None:
        return aggregate_series(pd.Series(dtype=float), "building_adjacency", num_quantiles)

    values = momepy.building_adjacency(contiguity, knn)
    focal_values = values.reindex(b.index)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(building_adjacency=focal_values), features_dir / "building_adjacency.gpkg")
    return aggregate_series(focal_values, "building_adjacency", num_quantiles)
