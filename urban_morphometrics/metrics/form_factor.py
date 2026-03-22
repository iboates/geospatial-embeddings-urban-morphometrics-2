"""Form factor metric.

surface / volume^(2/3), where surface = (perimeter × height) + area.
A 3D compactness measure using building height. Low values indicate compact
blocks; high values indicate thin or tall buildings.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("form_factor")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """surface / volume^(2/3) per building using resolved OSM heights."""
    b = ctx.buildings_with_height
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "form_factor", num_quantiles)
    values = momepy.form_factor(b, b["height"])
    if features_dir is not None:
        write_features(b[["geometry"]].assign(form_factor=values), features_dir / "form_factor.gpkg")
    return aggregate_series(values, "form_factor", num_quantiles)
