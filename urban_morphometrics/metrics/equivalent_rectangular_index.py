"""Equivalent rectangular index (ERI) metric.

Ratio comparing the building's area and perimeter to an equivalent rectangle.
High ERI → rectangular footprints (row houses, slabs); low ERI → complex
footprints (L-shapes, courtyards).
"""

from pathlib import Path

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("equivalent_rectangular_index")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """ERI per building — compares area and perimeter to an equivalent rectangle."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "equivalent_rectangular_index", num_quantiles)
    values = momepy.equivalent_rectangular_index(b)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(equivalent_rectangular_index=values), features_dir / "equivalent_rectangular_index.gpkg")
    return aggregate_series(values, "equivalent_rectangular_index", num_quantiles)
