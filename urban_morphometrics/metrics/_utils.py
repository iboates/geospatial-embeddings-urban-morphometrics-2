"""Shared utilities for metric computation."""

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union


def dissolve_touching(buildings_ea: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve buildings that share walls into unified structures.

    A small buffer (0.01 m) is applied before dissolving to ensure buildings
    that merely touch are merged. The result is exploded back into individual
    single-part polygons, one per merged structure.

    Args:
        buildings_ea: Buildings in equal-area CRS.

    Returns:
        GeoDataFrame of dissolved structures in the same CRS.
    """
    if buildings_ea.empty:
        return buildings_ea[["geometry"]].copy()

    merged = unary_union(buildings_ea.geometry.buffer(0.01))
    geoms = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    return gpd.GeoDataFrame(geometry=geoms, crs=buildings_ea.crs)


def empty_series(index=None) -> pd.Series:
    return pd.Series([], dtype=float) if index is None else pd.Series(dtype=float, index=index[:0])
