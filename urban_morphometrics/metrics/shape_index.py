"""Shape index metric.

sqrt(area / pi) / (0.5 * longest_axis). Measures how close the footprint shape
is to a circle. A perfect circle scores 1; lower values indicate elongated or
irregular shapes.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("shape_index")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """√(area/π) / (0.5 × longest_axis) per building — 1 for a perfect circle."""
    b = ctx.buildings_ea
    empty = pd.Series(dtype=float)
    if b.empty:
        return {
            **aggregate_series(empty, "shape_index", num_quantiles),
            **aggregate_series(empty, "shape_index_joined", num_quantiles),
        }
    values = momepy.shape_index(b)

    d = ctx.dissolved_buildings_ea
    joined_values = momepy.shape_index(d) if not d.empty else empty

    if features_dir is not None:
        write_features(b[["geometry"]].assign(shape_index=values), features_dir / "shape_index.gpkg")
        write_features(d[["geometry"]].assign(shape_index_joined=joined_values), features_dir / "shape_index_joined.gpkg")

    result = aggregate_series(values, "shape_index", num_quantiles)
    result.update(aggregate_series(joined_values, "shape_index_joined", num_quantiles))
    return result
