"""Proportional water cover metric.

Computes the fraction of the focal cell's area covered by water polygons
(OSM tag: natural=water). Values are in [0, 1].

Water polygons are clipped to the cell boundary before area is summed.
"""

from pathlib import Path

import geopandas as gpd

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.features import write_features


@register("water_cover")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Proportional cover of water (natural=water) within the focal cell."""
    cell_ea = ctx._cell_ea
    water = ctx.water_ea

    if water.empty:
        return {"water_proportion": 0.0}

    clipped_geoms = water.geometry.intersection(cell_ea)
    cell_area = cell_ea.area
    covered = clipped_geoms.area.sum()

    if features_dir is not None:
        clipped_gdf = gpd.GeoDataFrame(
            {"water_proportion": clipped_geoms.area / cell_area},
            geometry=clipped_geoms,
            crs=water.crs,
        ).loc[clipped_geoms.area > 0]
        write_features(clipped_gdf, features_dir / "water_proportion.gpkg")

    return {"water_proportion": min(covered / cell_area, 1.0)}
