"""Floor area metric.

Footprint area multiplied by the number of floors. Floor count is derived from
the OSM 'building:levels' tag; if absent, height / 3 m is used as a fallback.
All areas are in square metres (equal-area CRS).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features

_LEVELS_COL = "building:levels"
_METRES_PER_STOREY = 3.0


def _floor_counts(buildings) -> pd.Series:
    """Derive floor count per building from levels tag or height fallback.

    Uses fillna() with a Series rather than boolean-mask assignment to avoid
    a pandas edge case where a named boolean Series (name='building:levels')
    is misinterpreted as a label key instead of a mask.
    """
    if _LEVELS_COL in buildings.columns:
        levels = pd.to_numeric(buildings[_LEVELS_COL], errors="coerce").rename(None)
    else:
        levels = pd.Series(np.nan, index=buildings.index, dtype=float)

    if "height" in buildings.columns:
        height_derived = pd.to_numeric(buildings["height"], errors="coerce") / _METRES_PER_STOREY
        levels = levels.fillna(height_derived)

    return levels.fillna(1.0)


@register("floor_area")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Compute floor area statistics for the focal cell.

    Floor area = footprint area × number of floors, where floors come from
    OSM 'building:levels' or height / 3 m as fallback. Areas in square metres.
    """
    b = ctx.buildings_with_height
    if b.empty:
        return aggregate_series(pd.Series([], dtype=float), "floor_area", num_quantiles)

    footprint_area = b.geometry.area
    floors = _floor_counts(b)
    floor_area = footprint_area * floors
    if features_dir is not None:
        write_features(b[["geometry"]].assign(floor_area=floor_area), features_dir / "floor_area.gpkg")
    return aggregate_series(floor_area, "floor_area", num_quantiles)
