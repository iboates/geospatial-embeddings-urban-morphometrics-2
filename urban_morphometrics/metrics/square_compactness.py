"""Square compactness metric.

(4 * sqrt(area) / perimeter)^2. Measures how efficiently the perimeter encloses
area. A perfect square scores 1; lower values indicate less compact shapes.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("square_compactness")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """(4√area / perimeter)² per building — 1 for a perfect square."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "square_compactness", num_quantiles)
    values = momepy.square_compactness(b)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(square_compactness=values), features_dir / "square_compactness.gpkg")
    return aggregate_series(values, "square_compactness", num_quantiles)
