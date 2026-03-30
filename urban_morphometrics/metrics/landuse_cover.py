"""Proportional landuse cover metrics.

For each of the 11 tracked landuse types, computes the fraction of the focal
cell's area covered by that landuse type. Values are in [0, 1]; a cell with
no data for a type returns 0.0 for that type.

Landuse polygons are clipped to the cell boundary before area is summed so
that polygons extending beyond the cell are not double-counted.
"""

from pathlib import Path

import geopandas as gpd

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.features import write_features

_LANDUSE_TYPES = [
    "farmland",
    "residential",
    "grass",
    "forest",
    "meadow",
    "orchard",
    "farmyard",
    "industrial",
    "vineyard",
    "cemetery",
    "commercial",
]


@register("landuse_cover")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Proportional cover of each tracked landuse type within the focal cell."""
    cell_ea = ctx._cell_ea
    cell_area = cell_ea.area
    landuse = ctx.landuse_ea

    result = {}
    for lu_type in _LANDUSE_TYPES:
        col = f"landuse_{lu_type}_proportion"
        if landuse.empty or "landuse" not in landuse.columns:
            result[col] = 0.0
            continue
        subset = landuse[landuse["landuse"] == lu_type]
        if subset.empty:
            result[col] = 0.0
            continue
        clipped_geoms = subset.geometry.intersection(cell_ea)
        covered = clipped_geoms.area.sum()
        result[col] = min(covered / cell_area, 1.0)
        if features_dir is not None:
            clipped_gdf = gpd.GeoDataFrame(
                {col: clipped_geoms.area / cell_area},
                geometry=clipped_geoms,
                crs=landuse.crs,
            ).loc[clipped_geoms.area > 0]
            write_features(clipped_gdf, features_dir / f"{col}.gpkg")

    return result
