"""Courtyard index metric.

Ratio of courtyard (interior hole) area to total footprint area, computed on
dissolved touching structures. Only structures that actually contain courtyards
contribute; cells with no courtyards produce all-NaN aggregated statistics.
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics._utils import dissolve_touching
from urban_morphometrics.metrics.features import write_features


@register("courtyard_index")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Ratio of interior courtyard area to total footprint area per dissolved structure."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "courtyard_index", num_quantiles)
    dissolved = dissolve_touching(b)
    values = momepy.courtyard_index(dissolved)
    if features_dir is not None:
        write_features(dissolved[["geometry"]].assign(courtyard_index=values), features_dir / "courtyard_index.gpkg")
    return aggregate_series(values, "courtyard_index", num_quantiles)
