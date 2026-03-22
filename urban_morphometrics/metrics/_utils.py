"""Shared utilities for metric computation."""

from collections import defaultdict

import geopandas as gpd
import pandas as pd
from shapely.ops import linemerge, unary_union

_COORD_ROUND = 8


def _endpoints(geom):
    coords = list(geom.coords)
    s = (round(coords[0][0], _COORD_ROUND), round(coords[0][1], _COORD_ROUND))
    e = (round(coords[-1][0], _COORD_ROUND), round(coords[-1][1], _COORD_ROUND))
    return s, e


def remove_interstitial_nodes_preserving_oneway(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove degree-2 interstitial nodes from a highways GeoDataFrame while
    preserving oneway boundaries.

    A node is interstitial (safe to remove) only if it has exactly degree 2 in
    the full graph AND both adjacent edges share the same oneway value. Nodes at
    a junction between a oneway=True and a oneway=False edge are kept even when
    they are degree 2 — removing them would destroy the intersection and corrupt
    the directed graph.

    Uses union-find to identify chains of same-oneway edges connected through
    interstitial nodes, then merges each chain with shapely linemerge.

    Args:
        gdf: Highway GeoDataFrame with a boolean 'oneway' column and LineString
             geometries in any projected CRS.

    Returns:
        Cleaned GeoDataFrame with interstitial nodes dissolved. All rows have
        oneway=True or oneway=False (no mixed or null values from aggregation).
    """
    # Map each rounded endpoint coordinate to the edge indices that touch it.
    point_edges: dict[tuple, list] = defaultdict(list)
    for idx, row in gdf.iterrows():
        s, e = _endpoints(row.geometry)
        point_edges[s].append(idx)
        point_edges[e].append(idx)

    # A node is interstitial only when exactly 2 edges meet and both share the
    # same oneway value. A T-junction (≥3 edges) or a oneway/bidirectional
    # boundary (different oneway values) must be preserved.
    interstitial: set[tuple] = set()
    for pt, idxs in point_edges.items():
        if len(idxs) == 2:
            a, b = idxs
            if gdf.loc[a, "oneway"] == gdf.loc[b, "oneway"]:
                interstitial.add(pt)

    # Union-Find: join edges connected through interstitial nodes into chains.
    parent = {idx: idx for idx in gdf.index}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for pt in interstitial:
        a, b = point_edges[pt]
        union(a, b)

    # Group edge indices by component root.
    groups: dict[int, list] = defaultdict(list)
    for idx in gdf.index:
        groups[find(idx)].append(idx)

    # Build output: singleton edges unchanged, chains merged.
    geom_col = gdf.geometry.name
    rows = []
    for idxs in groups.values():
        if len(idxs) == 1:
            rows.append(gdf.loc[idxs[0]])
        else:
            merged_geom = linemerge([gdf.loc[i, geom_col] for i in idxs])
            row = gdf.loc[idxs[0]].copy()
            row[geom_col] = merged_geom
            rows.append(row)

    result = gpd.GeoDataFrame(rows, crs=gdf.crs).reset_index(drop=True)
    return result.set_geometry(geom_col)


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
