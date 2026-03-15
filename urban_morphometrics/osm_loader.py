"""Load OSM data for a study area from a .pbf file using QuackOSM."""

import logging
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from quackosm import convert_pbf_to_geodataframe

# QuackOSM spawns worker processes. On Linux/WSL2 the default start method may
# be "spawn", which requires a `if __name__ == "__main__"` guard in the caller.
# Setting it to "fork" avoids that requirement and is safe on Linux.
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("fork")

log = logging.getLogger(__name__)

_POLYGON_TYPES = {"Polygon", "MultiPolygon"}
_LINE_TYPES = {"LineString", "MultiLineString"}


def _keep_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    before = len(gdf)
    gdf = gdf[gdf.geometry.geom_type.isin(_POLYGON_TYPES)].copy()
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geometry.geom_type == "Polygon"]
    log.debug("  geometry filter: %d -> %d polygons", before, len(gdf))
    return gdf


def _keep_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    before = len(gdf)
    gdf = gdf[gdf.geometry.geom_type.isin(_LINE_TYPES)].copy()
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geometry.geom_type == "LineString"]
    log.debug("  geometry filter: %d -> %d linestrings", before, len(gdf))
    return gdf


@dataclass
class OsmData:
    buildings: gpd.GeoDataFrame
    highways: gpd.GeoDataFrame
    landuse: gpd.GeoDataFrame


def load_osm_data(pbf_path: Path, study_area_gdf: gpd.GeoDataFrame) -> OsmData:
    """Load buildings, highways, and landuse from a .pbf file clipped to the study area.

    Results are cached automatically by QuackOSM based on the pbf path and filters.

    Args:
        pbf_path: Local path to the .pbf OSM dump.
        study_area_gdf: Study area in WGS84. The union of all polygons is used as
            the geometry filter so only features overlapping the study area are loaded.

    Returns:
        OsmData with buildings, highways, and landuse GeoDataFrames.
    """
    geometry_filter = study_area_gdf.geometry.union_all()

    log.info("Loading buildings from OSM...")
    buildings = convert_pbf_to_geodataframe(
        pbf_path,
        tags_filter={"building": True, "building:levels": True, "height": True},
        geometry_filter=geometry_filter,
        keep_all_tags=False,
    )
    buildings = _keep_polygons(buildings)
    log.info("  %d buildings loaded", len(buildings))

    log.info("Loading highways from OSM...")
    highways = convert_pbf_to_geodataframe(
        pbf_path,
        tags_filter={"highway": True, "oneway": True},
        geometry_filter=geometry_filter,
        keep_all_tags=False,
    )
    highways = _keep_lines(highways)
    log.info("  %d highway segments loaded", len(highways))

    log.info("Loading landuse from OSM...")
    landuse = convert_pbf_to_geodataframe(
        pbf_path,
        tags_filter={"landuse": True},
        geometry_filter=geometry_filter,
        keep_all_tags=False,
    )
    landuse = _keep_polygons(landuse)
    log.info("  %d landuse features loaded", len(landuse))

    return OsmData(buildings=buildings, highways=highways, landuse=landuse)
