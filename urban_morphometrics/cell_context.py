"""Per-cell spatial context with lazy computation and Parquet-backed caching.

CellContext prepares all projected and filtered GeoDataFrames needed to compute
metrics for a single study area cell. Each property is computed once and cached
to disk so the pipeline can resume after interruption.
"""

import logging
from functools import cached_property
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

from urban_morphometrics.constants import VEHICLE_HIGHWAY_TYPES, PEDESTRIAN_HIGHWAY_TYPES
from urban_morphometrics.height import resolve_heights
from urban_morphometrics.oneway import apply_oneway
from urban_morphometrics.osm_loader import OsmData

log = logging.getLogger(__name__)


class CellContext:
    """Lazy, cached spatial context for a single study area cell.

    All properties are computed on first access and persisted to Parquet under
    cache_dir. Subsequent accesses (including across pipeline restarts) load
    from the cache instead of recomputing.

    All projected GeoDataFrames use one of the three provided CRS strings:
      - equal-area  (_ea): preserves area; used for most building metrics
      - equidistant (_ed): preserves distance; used for network metrics
      - conformal   (_cf): preserves angles/shape; used for shape metrics

    Neighbourhood properties combine the focal cell's features with features
    from a buffer zone outside the cell to avoid edge effects in graph-based
    metrics. The neighbourhood distance is in metres (applied in equal-area CRS).
    """

    def __init__(
        self,
        region_id,
        cell_geometry,
        osm_data: OsmData,
        neighbourhood_distance: float,
        equal_area_crs: str,
        equidistant_crs: str,
        conformal_crs: str,
        cache_dir: Path,
    ):
        self.region_id = region_id
        self._cell_geometry = cell_geometry
        self._osm_data = osm_data
        self._neighbourhood_distance = neighbourhood_distance
        self._ea_crs = equal_area_crs
        self._ed_crs = equidistant_crs
        self._cf_crs = conformal_crs
        self._cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.parquet"

    def _load_or_compute(self, name: str, compute_fn) -> gpd.GeoDataFrame:
        path = self._cache_path(name)
        if path.exists():
            return gpd.read_parquet(path)
        result = compute_fn()
        result.to_parquet(path)
        return result

    # ------------------------------------------------------------------
    # Internal derived geometries (not cached to disk, cheap to recompute)
    # ------------------------------------------------------------------

    @cached_property
    def _cell_series_wgs84(self) -> gpd.GeoSeries:
        return gpd.GeoSeries([self._cell_geometry], crs="EPSG:4326")

    @cached_property
    def _cell_ea(self):
        return self._cell_series_wgs84.to_crs(self._ea_crs).iloc[0]

    @cached_property
    def _cell_buffer_wgs84(self):
        """Cell buffered by neighbourhood_distance, back in WGS84."""
        buffer_ea = gpd.GeoSeries([self._cell_ea], crs=self._ea_crs).buffer(
            self._neighbourhood_distance
        )
        return buffer_ea.to_crs("EPSG:4326").iloc[0]

    @cached_property
    def _focal_building_index(self) -> pd.Index:
        """Index of OSM buildings that intersect this cell."""
        b = self._osm_data.buildings
        return b.index[b.intersects(self._cell_geometry)]

    @cached_property
    def _neighbourhood_building_index(self) -> pd.Index:
        """Index of OSM buildings inside the neighbourhood buffer but not the cell."""
        b = self._osm_data.buildings
        in_buffer = b.intersects(self._cell_buffer_wgs84)
        not_focal = ~b.index.isin(self._focal_building_index)
        return b.index[in_buffer & not_focal]

    @cached_property
    def _focal_highway_index(self) -> pd.Index:
        """Index of OSM highways that intersect this cell."""
        h = self._osm_data.highways
        return h.index[h.intersects(self._cell_geometry)]

    @cached_property
    def _neighbourhood_highway_index(self) -> pd.Index:
        """Index of OSM highways inside the neighbourhood buffer but not the cell."""
        h = self._osm_data.highways
        in_buffer = h.intersects(self._cell_buffer_wgs84)
        not_focal = ~h.index.isin(self._focal_highway_index)
        return h.index[in_buffer & not_focal]

    # ------------------------------------------------------------------
    # Buildings (focal cell)
    # ------------------------------------------------------------------

    @cached_property
    def buildings_ea(self) -> gpd.GeoDataFrame:
        """Focal buildings projected to equal-area CRS."""
        return self._load_or_compute(
            "buildings_ea",
            lambda: self._osm_data.buildings.loc[self._focal_building_index].to_crs(self._ea_crs),
        )

    @cached_property
    def buildings_ed(self) -> gpd.GeoDataFrame:
        """Focal buildings projected to equidistant CRS."""
        return self._load_or_compute(
            "buildings_ed",
            lambda: self._osm_data.buildings.loc[self._focal_building_index].to_crs(self._ed_crs),
        )

    @cached_property
    def buildings_cf(self) -> gpd.GeoDataFrame:
        """Focal buildings projected to conformal CRS."""
        return self._load_or_compute(
            "buildings_cf",
            lambda: self._osm_data.buildings.loc[self._focal_building_index].to_crs(self._cf_crs),
        )

    @cached_property
    def buildings_with_height(self) -> gpd.GeoDataFrame:
        """Focal buildings in equal-area CRS with a resolved numeric 'height' column.

        Height resolution priority: OSM 'height' tag → 'building:levels' * 3 m → 6 m default.
        """
        return self._load_or_compute(
            "buildings_with_height",
            lambda: resolve_heights(self.buildings_ea),
        )

    # ------------------------------------------------------------------
    # Highways (focal cell)
    # ------------------------------------------------------------------

    def _filter_highways(self, types: set, crs: str, index: pd.Index) -> gpd.GeoDataFrame:
        h = self._osm_data.highways.loc[index]
        h = h[h["highway"].isin(types)]
        h = apply_oneway(h)
        return h.to_crs(crs)

    @cached_property
    def vehicle_highways_ea(self) -> gpd.GeoDataFrame:
        """Focal vehicle highways in equal-area CRS with boolean 'oneway' column."""
        return self._load_or_compute(
            "vehicle_highways_ea",
            lambda: self._filter_highways(VEHICLE_HIGHWAY_TYPES, self._ea_crs, self._focal_highway_index),
        )

    @cached_property
    def vehicle_highways_ed(self) -> gpd.GeoDataFrame:
        """Focal vehicle highways in equidistant CRS with boolean 'oneway' column."""
        return self._load_or_compute(
            "vehicle_highways_ed",
            lambda: self._filter_highways(VEHICLE_HIGHWAY_TYPES, self._ed_crs, self._focal_highway_index),
        )

    @cached_property
    def vehicle_highways_cf(self) -> gpd.GeoDataFrame:
        """Focal vehicle highways in conformal CRS with boolean 'oneway' column."""
        return self._load_or_compute(
            "vehicle_highways_cf",
            lambda: self._filter_highways(VEHICLE_HIGHWAY_TYPES, self._cf_crs, self._focal_highway_index),
        )

    @cached_property
    def pedestrian_highways_ea(self) -> gpd.GeoDataFrame:
        """Focal pedestrian highways in equal-area CRS."""
        return self._load_or_compute(
            "pedestrian_highways_ea",
            lambda: self._filter_highways(PEDESTRIAN_HIGHWAY_TYPES, self._ea_crs, self._focal_highway_index),
        )

    @cached_property
    def pedestrian_highways_ed(self) -> gpd.GeoDataFrame:
        """Focal pedestrian highways in equidistant CRS."""
        return self._load_or_compute(
            "pedestrian_highways_ed",
            lambda: self._filter_highways(PEDESTRIAN_HIGHWAY_TYPES, self._ed_crs, self._focal_highway_index),
        )

    @cached_property
    def pedestrian_highways_cf(self) -> gpd.GeoDataFrame:
        """Focal pedestrian highways in conformal CRS."""
        return self._load_or_compute(
            "pedestrian_highways_cf",
            lambda: self._filter_highways(PEDESTRIAN_HIGHWAY_TYPES, self._cf_crs, self._focal_highway_index),
        )

    # ------------------------------------------------------------------
    # Landuse (focal cell)
    # ------------------------------------------------------------------

    @cached_property
    def landuse_ea(self) -> gpd.GeoDataFrame:
        """Focal landuse polygons in equal-area CRS."""
        return self._load_or_compute(
            "landuse_ea",
            lambda: self._osm_data.landuse.loc[
                self._osm_data.landuse.index[
                    self._osm_data.landuse.intersects(self._cell_geometry)
                ]
            ].to_crs(self._ea_crs),
        )

    # ------------------------------------------------------------------
    # Neighbourhood layers
    # ------------------------------------------------------------------

    @cached_property
    def neighbourhood_buildings(self) -> gpd.GeoDataFrame:
        """Buildings within neighbourhood_distance of the cell but outside it, in equal-area CRS."""
        return self._load_or_compute(
            "neighbourhood_buildings",
            lambda: self._osm_data.buildings.loc[self._neighbourhood_building_index].to_crs(self._ea_crs),
        )

    @cached_property
    def neighbourhood_vehicle_highways(self) -> gpd.GeoDataFrame:
        """Vehicle highways within neighbourhood_distance of the cell but outside it, in equal-area CRS."""
        return self._load_or_compute(
            "neighbourhood_vehicle_highways",
            lambda: self._filter_highways(VEHICLE_HIGHWAY_TYPES, self._ea_crs, self._neighbourhood_highway_index),
        )

    @cached_property
    def neighbourhood_pedestrian_highways(self) -> gpd.GeoDataFrame:
        """Pedestrian highways within neighbourhood_distance of the cell but outside it, in equal-area CRS."""
        return self._load_or_compute(
            "neighbourhood_pedestrian_highways",
            lambda: self._filter_highways(PEDESTRIAN_HIGHWAY_TYPES, self._ea_crs, self._neighbourhood_highway_index),
        )

    # ------------------------------------------------------------------
    # Focal + neighbourhood combined (for edge-effect-free metrics)
    # ------------------------------------------------------------------

    @cached_property
    def focal_plus_neighbourhood_buildings(self) -> gpd.GeoDataFrame:
        """Focal and neighbourhood buildings combined, in equal-area CRS."""
        return pd.concat([self.buildings_ea, self.neighbourhood_buildings])

    @cached_property
    def focal_plus_neighbourhood_vehicle_highways(self) -> gpd.GeoDataFrame:
        """Focal and neighbourhood vehicle highways combined, in equal-area CRS."""
        return pd.concat([self.vehicle_highways_ea, self.neighbourhood_vehicle_highways])

    @cached_property
    def focal_plus_neighbourhood_pedestrian_highways(self) -> gpd.GeoDataFrame:
        """Focal and neighbourhood pedestrian highways combined, in equal-area CRS."""
        return pd.concat([self.pedestrian_highways_ea, self.neighbourhood_pedestrian_highways])
