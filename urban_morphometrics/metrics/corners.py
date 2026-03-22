"""Corners metric.

Count of vertices where the interior angle deviates significantly from 180°.
Captures footprint complexity — more corners indicate more articulated shapes.
"""

from pathlib import Path
import warnings

import pandas as pd
import momepy

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("corners")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Number of significant corners (angle deviation > 10° from 180°) per building.

    Suppresses a known momepy RuntimeWarning: floating-point arithmetic can produce
    a cosine slightly outside [-1, 1] for collinear vertices, causing arccos to return
    NaN. Momepy's boolean logic then treats that vertex as a non-corner, which is the
    correct result. The fix (np.clip before arccos) belongs upstream in momepy.
    """
    b = ctx.buildings_ea
    empty = pd.Series(dtype=float)
    if b.empty:
        return {
            **aggregate_series(empty, "corners", num_quantiles),
            **aggregate_series(empty, "corners_joined", num_quantiles),
        }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in arccos", RuntimeWarning)
        values = momepy.corners(b)

    d = ctx.dissolved_buildings_ea
    if not d.empty:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in arccos", RuntimeWarning)
            joined_values = momepy.corners(d)
    else:
        joined_values = empty

    if features_dir is not None:
        write_features(b[["geometry"]].assign(corners=values), features_dir / "corners.gpkg")
        write_features(d[["geometry"]].assign(corners_joined=joined_values), features_dir / "corners_joined.gpkg")

    result = aggregate_series(values, "corners", num_quantiles)
    result.update(aggregate_series(joined_values, "corners_joined", num_quantiles))
    return result
