"""Elongation metric.

Ratio of the shorter to the longer side of the minimum bounding rectangle.
Values near 1 indicate compact/square footprints; low values indicate elongated
buildings such as row houses or linear strips.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("elongation")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Shorter/longer side ratio of the minimum bounding rectangle per building (0–1)."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "elongation", num_quantiles)
    values = momepy.elongation(b)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(elongation=values), features_dir / "elongation.gpkg")
    return aggregate_series(values, "elongation", num_quantiles)
