"""Courtyard area metric.

Adjacent buildings are dissolved into unified structures. Each interior ring
(hole) in the dissolved geometry is extracted as an individual courtyard polygon.
Statistics are computed over individual courtyard areas, plus a total count.
Uses equal-area CRS so areas are in square metres.

Requires neighbourhood context so buildings near the cell boundary are connected
to their neighbours across the boundary, avoiding false courtyard splits at the
cell edge.
"""

from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


def _extract_courtyard_areas(
    focal: gpd.GeoDataFrame, all_buildings: gpd.GeoDataFrame
) -> tuple[pd.Series, list]:
    """Dissolve touching buildings and extract interior ring areas and geometries.

    Dissolves *all* buildings (focal + neighbourhood) so courtyards that span
    the cell boundary are captured correctly, then keeps only courtyards whose
    centroid falls within a focal building footprint.

    Returns a tuple of (areas Series, list of courtyard Polygon geometries).
    """
    if all_buildings.empty:
        return pd.Series([], dtype=float), []

    dissolved = unary_union(all_buildings.geometry)
    geoms = dissolved.geoms if hasattr(dissolved, "geoms") else [dissolved]

    focal_union = unary_union(focal.geometry) if not focal.empty else None

    areas = []
    courtyard_geoms = []
    for geom in geoms:
        if not isinstance(geom, Polygon):
            continue
        for interior in geom.interiors:
            courtyard = Polygon(interior)
            if focal_union is not None and focal_union.intersects(courtyard):
                areas.append(courtyard.area)
                courtyard_geoms.append(courtyard)
    return pd.Series(areas, dtype=float), courtyard_geoms


@register("courtyard_area")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Compute courtyard area statistics for the focal cell.

    Dissolves touching buildings (focal + neighbourhood) into unified structures
    and measures the area of each interior courtyard ring that intersects a
    focal building
    All areas are in square metres (equal-area CRS).
    """
    b = ctx.buildings_ea
    all_b = ctx.focal_plus_neighbourhood_buildings
    courtyard_areas, courtyard_geoms = _extract_courtyard_areas(b, all_b)
    if features_dir is not None and courtyard_geoms:
        import geopandas as gpd
        courtyard_gdf = gpd.GeoDataFrame(
            {"courtyard_area": courtyard_areas.values},
            geometry=courtyard_geoms,
            crs=b.crs,
        )
        write_features(courtyard_gdf, features_dir / "courtyard_area.gpkg")
    result = aggregate_series(courtyard_areas, "courtyard_area", num_quantiles)
    return result
