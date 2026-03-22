"""Courtyard index metric.

Ratio of courtyard (interior hole) area to total footprint area, computed on
dissolved touching structures. Focal and neighbourhood buildings are dissolved
together so courtyards that span the cell boundary are captured correctly.
Only dissolved structures that intersect the focal cell and contain at least
one interior ring contribute to the statistics. Cells with no courtyards
produce all-NaN aggregated statistics.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import momepy
from shapely.ops import unary_union
from shapely.geometry import Polygon

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("courtyard_index")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Ratio of interior courtyard area to total footprint area, for structures with courtyards only."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "courtyard_index", num_quantiles)

    all_b = ctx.focal_plus_neighbourhood_buildings
    dissolved = unary_union(all_b.geometry)
    geoms = dissolved.geoms if hasattr(dissolved, "geoms") else [dissolved]

    structures = gpd.GeoDataFrame(
        geometry=[g for g in geoms if isinstance(g, Polygon)],
        crs=all_b.crs,
    )

    focal_union = unary_union(b.geometry)
    structures = structures[
        structures.intersects(focal_union) &
        structures.geometry.apply(lambda g: len(g.interiors) > 0)
    ]

    if structures.empty:
        return aggregate_series(pd.Series(dtype=float), "courtyard_index", num_quantiles)

    values = momepy.courtyard_index(structures)
    if features_dir is not None:
        write_features(
            structures[["geometry"]].assign(courtyard_index=values),
            features_dir / "courtyard_index.gpkg",
        )
    return aggregate_series(values, "courtyard_index", num_quantiles)
