"""Tests for street graph construction and focal-node extraction."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from urban_morphometrics.street_graph import (
    build_vehicle_graph,
    build_pedestrian_graph,
    focal_nodes_series,
    nodes_gdf,
)


def _t_junction(crs="EPSG:3395"):
    """T-junction: three segments meeting at (50,0).

    Nodes: (0,0)—deg1, (50,0)—deg3, (100,0)—deg1, (50,100)—deg1
    No degree-2 nodes, so remove_false_nodes leaves all 4 nodes intact.
    """
    return gpd.GeoDataFrame(
        {"highway": ["residential"] * 3, "oneway": [False] * 3},
        geometry=[
            LineString([(0, 0), (50, 0)]),
            LineString([(50, 0), (100, 0)]),
            LineString([(50, 0), (50, 100)]),
        ],
        crs=crs,
    )


def _single_segment(crs="EPSG:3395"):
    return gpd.GeoDataFrame(
        {"highway": ["residential"], "oneway": [False]},
        geometry=[LineString([(0, 0), (100, 0)])],
        crs=crs,
    )


class TestBuildVehicleGraph:
    def test_returns_none_for_empty(self):
        empty = gpd.GeoDataFrame(
            {"highway": [], "oneway": []},
            geometry=gpd.GeoSeries([], crs="EPSG:3395"),
        )
        assert build_vehicle_graph(empty) is None

    def test_directed_graph(self):
        import networkx as nx
        G = build_vehicle_graph(_t_junction())
        assert G is not None
        assert isinstance(G, nx.MultiDiGraph)

    def test_correct_node_count(self):
        G = build_vehicle_graph(_t_junction())
        assert G.number_of_nodes() == 4

    def test_oneway_single_direction(self):
        gdf = gpd.GeoDataFrame(
            {"highway": ["residential"], "oneway": [True]},
            geometry=[LineString([(0, 0), (100, 0)])],
            crs="EPSG:3395",
        )
        G = build_vehicle_graph(gdf)
        assert G.number_of_edges() == 1

    def test_twoway_both_directions(self):
        G = build_vehicle_graph(_single_segment())
        assert G.number_of_edges() == 2


class TestBuildPedestrianGraph:
    def test_returns_none_for_empty(self):
        empty = gpd.GeoDataFrame(
            {"highway": [], "oneway": []},
            geometry=gpd.GeoSeries([], crs="EPSG:3395"),
        )
        assert build_pedestrian_graph(empty) is None

    def test_undirected_graph(self):
        import networkx as nx
        G = build_pedestrian_graph(_t_junction())
        assert G is not None
        assert isinstance(G, nx.MultiGraph)

    def test_correct_node_count(self):
        G = build_pedestrian_graph(_t_junction())
        assert G.number_of_nodes() == 4


class TestNodesGdf:
    def test_returns_geodataframe(self):
        G = build_pedestrian_graph(_t_junction())
        nodes = nodes_gdf(G)
        assert isinstance(nodes, gpd.GeoDataFrame)
        assert len(nodes) == 4

    def test_has_geometry_column(self):
        G = build_pedestrian_graph(_t_junction())
        nodes = nodes_gdf(G)
        assert "geometry" in nodes.columns


class TestFocalNodesSeries:
    def test_filters_to_cell(self):
        import momepy
        G = build_pedestrian_graph(_t_junction())
        G = momepy.node_degree(G)
        # Generous box that strictly contains the bottom two nodes
        # (0,0) and (50,0) and (100,0) but not (50,100)
        cell = box(-1, -1, 101, 50)
        series = focal_nodes_series(G, "degree", cell)
        assert len(series) == 3

    def test_empty_for_disjoint_cell(self):
        import momepy
        G = build_pedestrian_graph(_t_junction())
        G = momepy.node_degree(G)
        cell = box(200, 200, 300, 300)
        series = focal_nodes_series(G, "degree", cell)
        assert len(series) == 0

    def test_missing_attr_returns_empty(self):
        G = build_pedestrian_graph(_t_junction())
        cell = box(-1, -1, 200, 200)
        series = focal_nodes_series(G, "nonexistent_attr", cell)
        assert len(series) == 0

    def test_values_match_expected_degree(self):
        import momepy
        G = build_pedestrian_graph(_t_junction())
        G = momepy.node_degree(G)
        # All 4 nodes in a large cell
        cell = box(-1, -1, 200, 200)
        series = focal_nodes_series(G, "degree", cell)
        assert len(series) == 4
        assert sorted(series.tolist()) == [1, 1, 1, 3]
