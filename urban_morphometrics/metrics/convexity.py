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
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "convexity", num_quantiles)
    values = momepy.convexity(b)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(convexity=values), features_dir / "convexity.gpkg")
    return aggregate_series(values, "convexity", num_quantiles)
