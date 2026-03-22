"""Courtyards metric.

Total number of courtyards in the focal cell. All buildings (focal +
neighbourhood) are dissolved into superstructures; superstructures that
intersect the focal cell are kept, and their interior ring counts are summed.

Requires neighbourhood context so buildings near the cell boundary are connected
to their neighbours across the boundary, avoiding false courtyard splits at the
cell edge.
"""

from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.features import write_features


@register("courtyards")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Total courtyard count for the focal cell."""
    b = ctx.buildings_ea
    if b.empty:
        return {"courtyards_count": 0}

    all_b = ctx.focal_plus_neighbourhood_buildings
    dissolved = unary_union(all_b.geometry)
    geoms = dissolved.geoms if hasattr(dissolved, "geoms") else [dissolved]

    structures = gpd.GeoDataFrame(
        geometry=[g for g in geoms if isinstance(g, Polygon)],
        crs=all_b.crs,
    )

    focal_union = unary_union(b.geometry)
    structures = structures[structures.intersects(focal_union)]

    structures["courtyards"] = structures.geometry.apply(lambda g: len(g.interiors))

    if features_dir is not None:
        write_features(structures[["geometry", "courtyards"]], features_dir / "courtyards.gpkg")

    return {"courtyards_count": int(structures["courtyards"].sum())}
