"""Shared utilities for metric computation."""

from collections import defaultdict

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree

_COORD_ROUND = 8


def _endpoints(geom):
    coords = list(geom.coords)
    s = (round(coords[0][0], _COORD_ROUND), round(coords[0][1], _COORD_ROUND))
    e = (round(coords[-1][0], _COORD_ROUND), round(coords[-1][1], _COORD_ROUND))
    return s, e


def _cut_line(line: LineString, distances: list[float]) -> list[LineString]:
    """Split a LineString at the given distances (metres) along it.

    Distances outside (0, line.length) and duplicates closer than 1e-6 m are
    ignored. Returns a list of LineString pieces; if no valid cuts exist the
    original line is returned as a single-element list.
    """
    cuts = sorted(d for d in set(distances) if 0.0 < d < line.length)
    deduped: list[float] = []
    for d in cuts:
        if not deduped or d - deduped[-1] > 1e-6:
            deduped.append(d)
    if not deduped:
        return [line]

    # Strip Z so all coords are (x, y)
    coords = [c[:2] for c in line.coords]

    # Cumulative distances at each vertex
    cum = [0.0]
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        cum.append(cum[-1] + (dx * dx + dy * dy) ** 0.5)

    pieces: list[LineString] = []
    current = [coords[0]]
    seg_i = 0

    for cut_d in deduped:
        # Walk forward through segments until the segment containing cut_d
        while seg_i < len(coords) - 2 and cum[seg_i + 1] <= cut_d:
            seg_i += 1
            current.append(coords[seg_i])

        seg_span = cum[seg_i + 1] - cum[seg_i]
        if seg_span < 1e-12:
            continue  # degenerate segment; skip
        t = (cut_d - cum[seg_i]) / seg_span
        cx = coords[seg_i][0] + t * (coords[seg_i + 1][0] - coords[seg_i][0])
        cy = coords[seg_i][1] + t * (coords[seg_i + 1][1] - coords[seg_i][1])
        split_pt = (cx, cy)

        current.append(split_pt)
        if len(current) >= 2:
            pieces.append(LineString(current))
        current = [split_pt]

    # Remaining vertices from seg_i+1 to end
    current.extend(coords[seg_i + 1 :])
    if len(current) >= 2:
        pieces.append(LineString(current))

    return pieces if pieces else [line]


def split_lines_at_endpoints(gdf: gpd.GeoDataFrame, tolerance: float = 0.1) -> gpd.GeoDataFrame:
    """Split line segments wherever an endpoint of another line falls on them.

    Repairs T-junction topology: when a street endpoint lands on the interior
    of another street (within *tolerance* CRS units) the second street is split
    at that point so the resulting graph correctly represents the intersection.

    The *tolerance* controls both the snap search radius and the minimum distance
    from an existing endpoint before a split is triggered (endpoints that are
    already shared are left alone).

    All non-geometry columns are inherited by every piece produced by a split.

    Args:
        gdf:       Highway GeoDataFrame with LineString geometries.
        tolerance: Maximum distance (CRS units, metres for projected CRS) from
                   an endpoint to a candidate line for a split to be triggered.

    Returns:
        GeoDataFrame with split rows inserted; row count ≥ len(gdf).
    """
    geometries = list(gdf.geometry)
    tree = STRtree(geometries)
    geom_col = gdf.geometry.name

    # split_distances[positional_index] = list of distances along that line
    split_distances: dict[int, list[float]] = defaultdict(list)

    for src_idx, src_geom in enumerate(geometries):
        src_coords = list(src_geom.coords)
        endpoints = [Point(src_coords[0][:2]), Point(src_coords[-1][:2])]

        for ep in endpoints:
            for cand_idx in tree.query(ep.buffer(tolerance)):
                if cand_idx == src_idx:
                    continue
                cand_geom = geometries[cand_idx]
                if ep.distance(cand_geom) > tolerance:
                    continue

                # Skip if this endpoint already coincides with an endpoint of the
                # candidate (shared node — no split needed)
                cand_coords = list(cand_geom.coords)
                if ep.distance(Point(cand_coords[0][:2])) <= tolerance:
                    continue
                if ep.distance(Point(cand_coords[-1][:2])) <= tolerance:
                    continue

                split_distances[cand_idx].append(cand_geom.project(ep))

    if not split_distances:
        return gdf

    rows = []
    for pos_idx in range(len(gdf)):
        row = gdf.iloc[pos_idx]
        if pos_idx not in split_distances:
            rows.append(row)
        else:
            for piece in _cut_line(row[geom_col], split_distances[pos_idx]):
                new_row = row.copy()
                new_row[geom_col] = piece
                rows.append(new_row)

    result = gpd.GeoDataFrame(rows, crs=gdf.crs).reset_index(drop=True)
    return result.set_geometry(geom_col)


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
