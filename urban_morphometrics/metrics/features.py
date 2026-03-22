"""Helpers for exporting per-feature metric values as GeoPackage files."""

from pathlib import Path

import geopandas as gpd


def write_features(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Write *gdf* to *path* as a GeoPackage, silently skipping if empty."""
    if gdf is None or gdf.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")
