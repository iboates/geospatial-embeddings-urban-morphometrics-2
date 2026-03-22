"""Compactness-weighted axis metric.

d × (4/π − 16·area / perimeter²), where d is the longest axis length.
Combines longest axis length with compactness to measure how efficiently a
polygon fills space relative to its principal axes. Computed twice:
- `compactness_weighted_axis`: per raw individual building
- `compactness_weighted_axis_joined`: per dissolved structure (touching buildings merged)
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("compactness_weighted_axis")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Longest-axis-weighted compactness per raw building and per dissolved structure."""
    b = ctx.buildings_ea
    empty = pd.Series(dtype=float)
    if b.empty:
        return {
            **aggregate_series(empty, "compactness_weighted_axis", num_quantiles),
            **aggregate_series(empty, "compactness_weighted_axis_joined", num_quantiles),
        }

    raw_values = momepy.compactness_weighted_axis(b)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(compactness_weighted_axis=raw_values), features_dir / "compactness_weighted_axis.gpkg")
    result = aggregate_series(raw_values, "compactness_weighted_axis", num_quantiles)

    dissolved = ctx.dissolved_buildings_ea
    joined_values = momepy.compactness_weighted_axis(dissolved) if not dissolved.empty else pd.Series(dtype=float)
    if features_dir is not None:
        write_features(dissolved[["geometry"]].assign(compactness_weighted_axis_joined=joined_values), features_dir / "compactness_weighted_axis_joined.gpkg")
    result.update(aggregate_series(joined_values, "compactness_weighted_axis_joined", num_quantiles))

    return result
