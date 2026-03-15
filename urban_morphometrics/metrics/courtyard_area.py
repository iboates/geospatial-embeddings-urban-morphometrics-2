"""Courtyard area metric.

Adjacent buildings are dissolved into unified structures. Each interior ring
(hole) in the dissolved geometry is extracted as an individual courtyard polygon.
Statistics are computed over individual courtyard areas, plus a total count.
Uses equal-area CRS so areas are in square metres.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


def _extract_courtyard_areas(buildings_ea: gpd.GeoDataFrame) -> pd.Series:
    """Dissolve touching buildings and extract interior ring areas."""
    if buildings_ea.empty:
        return pd.Series([], dtype=float)

    dissolved = unary_union(buildings_ea.geometry)
    areas = []
    geoms = dissolved.geoms if hasattr(dissolved, "geoms") else [dissolved]
    for geom in geoms:
        if not isinstance(geom, Polygon):
            continue
        for interior in geom.interiors:
            areas.append(Polygon(interior).area)
    return pd.Series(areas, dtype=float)


@register("courtyard_area")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Compute courtyard area statistics for the focal cell.

    Dissolves touching buildings into unified structures and measures the area
    of each interior courtyard ring. Also records the total courtyard count.
    All areas are in square metres (equal-area CRS).
    """
    courtyard_areas = _extract_courtyard_areas(ctx.buildings_ea)
    result = aggregate_series(courtyard_areas, "courtyard_area", num_quantiles)
    result["courtyard_area_count"] = len(courtyard_areas)
    return result
