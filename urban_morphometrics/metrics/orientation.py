"""Orientation metric.

Deviation of each building's longest axis from cardinal directions (0°–45° range).
Low mean orientation → grid-aligned urban fabric; high values → organic layout.
No neighbourhood context needed — computed per individual building footprint.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("orientation")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Deviation of each building's longest axis from cardinal directions (0°–45°)."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "orientation", num_quantiles)
    values = momepy.orientation(b)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(orientation=values), features_dir / "orientation.gpkg")
    return aggregate_series(values, "orientation", num_quantiles)
