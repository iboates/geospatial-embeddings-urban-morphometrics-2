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
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "circular_compactness", num_quantiles)
    values = momepy.circular_compactness(b)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(circular_compactness=values), features_dir / "circular_compactness.gpkg")
    return aggregate_series(values, "circular_compactness", num_quantiles)
