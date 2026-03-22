"""Circular compactness metric.

Ratio of building area to the area of its minimum bounding circle.
Values near 1 indicate circular footprints; lower values indicate irregular shapes.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("circular_compactness")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Ratio of building area to its minimum bounding circle area (0–1)."""
    b = ctx.buildings_ea
    empty = pd.Series(dtype=float)
    if b.empty:
        return {
            **aggregate_series(empty, "circular_compactness", num_quantiles),
            **aggregate_series(empty, "circular_compactness_joined", num_quantiles),
        }
    values = momepy.circular_compactness(b)

    d = ctx.dissolved_buildings_ea
    joined_values = momepy.circular_compactness(d) if not d.empty else empty

    if features_dir is not None:
        write_features(b[["geometry"]].assign(circular_compactness=values), features_dir / "circular_compactness.gpkg")
        write_features(d[["geometry"]].assign(circular_compactness_joined=joined_values), features_dir / "circular_compactness_joined.gpkg")

    result = aggregate_series(values, "circular_compactness", num_quantiles)
    result.update(aggregate_series(joined_values, "circular_compactness_joined", num_quantiles))
    return result
