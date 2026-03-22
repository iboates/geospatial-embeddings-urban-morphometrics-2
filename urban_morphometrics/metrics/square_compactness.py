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
    empty = pd.Series(dtype=float)
    if b.empty:
        return {
            **aggregate_series(empty, "square_compactness", num_quantiles),
            **aggregate_series(empty, "square_compactness_joined", num_quantiles),
        }
    values = momepy.square_compactness(b)

    d = ctx.dissolved_buildings_ea
    joined_values = momepy.square_compactness(d) if not d.empty else empty

    if features_dir is not None:
        write_features(b[["geometry"]].assign(square_compactness=values), features_dir / "square_compactness.gpkg")
        write_features(d[["geometry"]].assign(square_compactness_joined=joined_values), features_dir / "square_compactness_joined.gpkg")

    result = aggregate_series(values, "square_compactness", num_quantiles)
    result.update(aggregate_series(joined_values, "square_compactness_joined", num_quantiles))
    return result
