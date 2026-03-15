"""Tests for OSM oneway tag parsing."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from urban_morphometrics.oneway import parse_oneway, apply_oneway


class TestParseOneway:
    def test_yes(self):
        assert parse_oneway("yes", "residential") is True

    def test_true(self):
        assert parse_oneway("true", "residential") is True

    def test_one(self):
        assert parse_oneway("1", "residential") is True

    def test_no(self):
        assert parse_oneway("no", "residential") is False

    def test_false(self):
        assert parse_oneway("false", "residential") is False

    def test_zero(self):
        assert parse_oneway("0", "residential") is False

    def test_minus_one(self):
        # reverse oneway still counts as one-way
        assert parse_oneway("-1", "residential") is True

    def test_none_residential(self):
        assert parse_oneway(None, "residential") is False

    def test_nan_residential(self):
        assert parse_oneway(float("nan"), "residential") is False

    def test_motorway_default(self):
        assert parse_oneway(None, "motorway") is True

    def test_motorway_link_default(self):
        assert parse_oneway(None, "motorway_link") is True

    def test_roundabout_default(self):
        assert parse_oneway(None, "residential", junction_val="roundabout") is True

    def test_roundabout_overrides_no(self):
        # explicit "no" beats the roundabout default
        assert parse_oneway("no", "residential", junction_val="roundabout") is False


class TestApplyOneway:
    def _make_gdf(self, oneway_vals, highway_vals):
        return gpd.GeoDataFrame(
            {"highway": highway_vals, "oneway": oneway_vals},
            geometry=[LineString([(0, 0), (1, 0)])] * len(oneway_vals),
            crs="EPSG:4326",
        )

    def test_adds_boolean_column(self):
        gdf = self._make_gdf(["yes", "no"], ["residential", "residential"])
        result = apply_oneway(gdf)
        assert result["oneway"].tolist() == [True, False]

    def test_original_unchanged(self):
        gdf = self._make_gdf(["yes"], ["residential"])
        original_val = gdf["oneway"].iloc[0]
        apply_oneway(gdf)
        assert gdf["oneway"].iloc[0] == original_val  # copy, not in-place

    def test_motorway_no_tag(self):
        gdf = self._make_gdf([None], ["motorway"])
        result = apply_oneway(gdf)
        assert result["oneway"].iloc[0]
