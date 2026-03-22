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
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "rectangularity", num_quantiles)
    values = momepy.rectangularity(b)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(rectangularity=values), features_dir / "rectangularity.gpkg")
    return aggregate_series(values, "rectangularity", num_quantiles)
