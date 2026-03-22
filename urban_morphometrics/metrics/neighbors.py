"""Neighbours metric.

Number of neighbours for each building's morphological tessellation cell,
measured by queen contiguity on the tessellation. Captures how many buildings
surround each building in the local urban fabric.

Requires neighbourhood context so the tessellation is not clipped at the cell
boundary, which would produce artificially small tessellation cells for buildings
near the edge.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("neighbors")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Number of queen-contiguous tessellation neighbours per focal building."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "neighbors", num_quantiles)

    tess = ctx.tessellation
    queen = ctx.tessellation_queen_graph
    if tess is None or tess.empty or queen is None:
        return aggregate_series(pd.Series(dtype=float), "neighbors", num_quantiles)

    values = momepy.neighbors(tess, queen)
    focal_values = values.reindex(b.index)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(neighbors=focal_values), features_dir / "neighbors.gpkg")
    return aggregate_series(focal_values, "neighbors", num_quantiles)
