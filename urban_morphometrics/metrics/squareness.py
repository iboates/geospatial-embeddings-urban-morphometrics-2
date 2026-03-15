"""Squareness metric.

Mean deviation of corner angles from 90°. Low values indicate buildings with
predominantly right-angle corners, typical of modern rectilinear construction.
High values indicate organic or irregular footprints.
"""

import warnings
import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("squareness")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
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
        return aggregate_series(momepy.squareness(b), "squareness", num_quantiles)
