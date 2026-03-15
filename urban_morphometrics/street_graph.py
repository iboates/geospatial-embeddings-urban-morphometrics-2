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
    metric attributes that have been added to the graph by momepy metric functions.
    """
    nodes, _ = momepy.nx_to_gdf(graph, points=True, lines=False)
    return nodes


def focal_node_ids(graph, cell_geom):
    """Return the list of node IDs (graph node keys) that lie within cell_geom.

    cell_geom must be a Shapely geometry in the same CRS as the graph nodes.
    Node positions are stored as 'x'/'y' attributes on each node by momepy.gdf_to_nx.
    """
    from shapely.geometry import Point

    return [
        node
        for node, data in graph.nodes(data=True)
        if cell_geom.contains(Point(data["x"], data["y"]))
    ]
