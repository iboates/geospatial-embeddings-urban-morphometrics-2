"""Convexity metric.

Ratio of building area to its convex hull area. Values near 1 mean the footprint
has no concavities; lower values indicate L-shapes, courtyards, or concave forms.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("convexity")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Ratio of building area to its convex hull area (0–1)."""
    b = ctx.buildings_ea
    empty = pd.Series(dtype=float)
    if b.empty:
        return {
            **aggregate_series(empty, "convexity", num_quantiles),
            **aggregate_series(empty, "convexity_joined", num_quantiles),
        }
    values = momepy.convexity(b)

    d = ctx.dissolved_buildings_ea
    joined_values = momepy.convexity(d) if not d.empty else empty

    if features_dir is not None:
        write_features(b[["geometry"]].assign(convexity=values), features_dir / "convexity.gpkg")
        write_features(d[["geometry"]].assign(convexity_joined=joined_values), features_dir / "convexity_joined.gpkg")

    result = aggregate_series(values, "convexity", num_quantiles)
    result.update(aggregate_series(joined_values, "convexity_joined", num_quantiles))
    return result
