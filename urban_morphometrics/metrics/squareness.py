"""Squareness metric.

Mean deviation of corner angles from 90°. Low values indicate buildings with
predominantly right-angle corners, typical of modern rectilinear construction.
High values indicate organic or irregular footprints.
"""

from pathlib import Path
import warnings

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("squareness")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Mean deviation of corner angles from 90° per building (degrees).

    Suppresses a known momepy RuntimeWarning: floating-point arithmetic can produce
    a cosine slightly outside [-1, 1] for collinear vertices, causing arccos to return
    NaN. Momepy's boolean logic then treats that vertex as a non-corner, which is the
    correct result. The fix (np.clip before arccos) belongs upstream in momepy.
    """
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "squareness", num_quantiles)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in arccos", RuntimeWarning)
        values = momepy.squareness(b)
    if features_dir is not None:
        write_features(b[["geometry"]].assign(squareness=values), features_dir / "squareness.gpkg")
    return aggregate_series(values, "squareness", num_quantiles)
