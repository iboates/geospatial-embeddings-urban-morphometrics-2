"""Perimeter wall metric.

Two variants:
- perimeter_wall_individual: the outer perimeter of each building individually
  (geometry.length in equidistant CRS).
- perimeter_wall_joined: buildings that share walls are dissolved into unified
  structures; one perimeter value per structure (not per building). Aggregated
  at structure level to avoid inflating statistics when multiple buildings share
  the same structure.

Both use equidistant CRS. Values are in metres.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.geometry import Polygon

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series
from urban_morphometrics.metrics.features import write_features


@register("perimeter_wall")
def compute(ctx: CellContext, num_quantiles: int, features_dir: Path | None = None) -> dict:
    """Perimeter of individual buildings and dissolved joined structures (metres)."""
    b = ctx.buildings_ed
    if b.empty:
        result = aggregate_series(pd.Series(dtype=float), "perimeter_wall_individual", num_quantiles)
        result.update(aggregate_series(pd.Series(dtype=float), "perimeter_wall_joined", num_quantiles))
        return result

    individual = b.geometry.length

    dissolved = unary_union(b.geometry)
    geoms = list(dissolved.geoms) if hasattr(dissolved, "geoms") else [dissolved]
    structures = gpd.GeoDataFrame(
        geometry=[g for g in geoms if isinstance(g, Polygon)],
        crs=b.crs,
    )
    joined = structures.geometry.length

    if features_dir is not None:
        write_features(
            b[["geometry"]].assign(perimeter_wall_individual=individual),
            features_dir / "perimeter_wall_individual.gpkg",
        )
        write_features(
            structures[["geometry"]].assign(perimeter_wall_joined=joined),
            features_dir / "perimeter_wall_joined.gpkg",
        )

    result = aggregate_series(individual, "perimeter_wall_individual", num_quantiles)
    result.update(aggregate_series(joined, "perimeter_wall_joined", num_quantiles))
    return result
