"""Street network graph construction for connectivity metrics.

Converts projected highway GeoDataFrames into NetworkX primal graphs
(endpoints = nodes, street segments = edges) suitable for momepy's
network analysis functions.

Two graph variants are produced per cell:
  - vehicle_graph: directed MultiDiGraph respecting OSM oneway tags.
  - pedestrian_graph: undirected MultiGraph treating all segments as
    bidirectional.

Both are built from focal + neighbourhood highways (in equidistant CRS)
to avoid edge effects on nodes near the cell boundary. Step 10 metrics
filter results back to focal nodes before aggregating.

Note: momepy.remove_false_nodes is used to clean up degree-2 nodes
(points in the middle of a street that are not true intersections). This
function is deprecated in momepy 0.11 in favour of neatnet, but the
underlying logic is unchanged and it continues to work correctly.
"""

import logging
import warnings

import momepy
import pandas as pd

log = logging.getLogger(__name__)


def build_vehicle_graph(highways_gdf):
    """Build a directed primal NetworkX graph from a vehicle highways GeoDataFrame.

    Edges respect the boolean 'oneway' column: one-way segments produce a single
    directed edge; bidirectional segments produce edges in both directions.

    Returns None if the GeoDataFrame is empty.
    """
    if highways_gdf.empty:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="momepy")
        cleaned = momepy.remove_false_nodes(highways_gdf)
    return momepy.gdf_to_nx(cleaned, directed=True, oneway_column="oneway")


def build_pedestrian_graph(highways_gdf):
    """Build an undirected primal NetworkX graph from a pedestrian highways GeoDataFrame.

    All segments are treated as bidirectional regardless of any oneway tag.

    Returns None if the GeoDataFrame is empty.
    """
    if highways_gdf.empty:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="momepy")
        cleaned = momepy.remove_false_nodes(highways_gdf)
    return momepy.gdf_to_nx(cleaned, directed=False)


def nodes_gdf(graph):
    """Extract node positions and attributes from a momepy primal graph as a GeoDataFrame.

    Returns a GeoDataFrame with one row per node, a Point geometry column, and any
    metric attributes added to the graph by momepy metric functions.
    momepy.nx_to_gdf always returns a (nodes, edges) tuple; we take only nodes.
    """
    nodes, _ = momepy.nx_to_gdf(graph)
    return nodes


def focal_nodes_series(graph, attr, cell_geom) -> pd.Series:
    """Extract a per-node metric attribute for nodes within the focal cell.

    Calls momepy.nx_to_gdf to materialise all node attributes, then spatially
    filters to nodes whose Point geometry lies within cell_geom (in the same
    CRS as the graph). Returns an empty Series if no focal nodes are found or
    the attribute is absent.

    Args:
        graph:      NetworkX graph with the attribute already set on nodes.
        attr:       Node attribute name to extract (e.g. 'degree', 'closeness').
        cell_geom:  Shapely geometry in the same projected CRS as the graph nodes.
    """
    nodes = nodes_gdf(graph)
    if attr not in nodes.columns:
        return pd.Series(dtype=float)
    mask = nodes.geometry.within(cell_geom)
    return nodes.loc[mask, attr].reset_index(drop=True)
