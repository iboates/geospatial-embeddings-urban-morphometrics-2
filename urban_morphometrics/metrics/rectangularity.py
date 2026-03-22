"""Rectangularity metric.

Ratio of building area to its minimum rotated bounding rectangle area.
Values near 1 indicate rectangular footprints.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("rectangularity")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Ratio of building area to its minimum rotated bounding rectangle (0–1)."""
    b = ctx.buildings_ea
    empty = pd.Series(dtype=float)
    if b.empty:
        return {
            **aggregate_series(empty, "rectangularity", num_quantiles),
            **aggregate_series(empty, "rectangularity_joined", num_quantiles),
        }
    values = momepy.rectangularity(b)

    d = ctx.dissolved_buildings_ea
    joined_values = momepy.rectangularity(d) if not d.empty else empty

    if features_dir is not None:
        write_features(b[["geometry"]].assign(rectangularity=values), features_dir / "rectangularity.gpkg")
        write_features(d[["geometry"]].assign(rectangularity_joined=joined_values), features_dir / "rectangularity_joined.gpkg")

    result = aggregate_series(values, "rectangularity", num_quantiles)
    result.update(aggregate_series(joined_values, "rectangularity_joined", num_quantiles))
    return result
