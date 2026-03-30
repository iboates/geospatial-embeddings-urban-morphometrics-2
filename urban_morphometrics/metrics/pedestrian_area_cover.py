"""Proportional pedestrian area cover metric.

Computes the fraction of the focal cell's area covered by pedestrian area
polygons (OSM tag: highway=pedestrian on closed ways/areas, i.e. plazas and
pedestrian zones). Values are in [0, 1].

Pedestrian area polygons are clipped to the cell boundary before area is summed.
"""

from pathlib import Path

import geopandas as gpd

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.features import write_features


@register("pedestrian_area_cover")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Proportional cover of pedestrian areas (highway=pedestrian polygons) within the focal cell."""
    cell_ea = ctx._cell_ea
    ped_areas = ctx.pedestrian_areas_ea

    if ped_areas.empty:
        return {"pedestrian_area_proportion": 0.0}

    clipped_geoms = ped_areas.geometry.intersection(cell_ea)
    cell_area = cell_ea.area
    covered = clipped_geoms.area.sum()

    if features_dir is not None:
        clipped_gdf = gpd.GeoDataFrame(
            {"pedestrian_area_proportion": clipped_geoms.area / cell_area},
            geometry=clipped_geoms,
            crs=ped_areas.crs,
        ).loc[clipped_geoms.area > 0]
        write_features(clipped_gdf, features_dir / "pedestrian_area_proportion.gpkg")

    return {"pedestrian_area_proportion": min(covered / cell_area, 1.0)}
