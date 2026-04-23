"""
Handles the full embedding workflow:
  1. Load OSM features for train / dev / test regions
  2. Fit the embedder on train data (if required)
  3. Transform all three splits into embedding DataFrames

Separating this from the main run script keeps things reusable and testable.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import geopandas as gpd
import pandas as pd
from srai.h3 import ring_buffer_h3_regions_gdf
from srai.joiners import IntersectionJoiner
from srai.loaders.osm_loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER
from srai.neighbourhoods.h3_neighbourhood import H3Neighbourhood

from urban_morphometrics.embedding.embedders.embedder_factory import requires_fit
from urban_morphometrics.main import compute_urban_morphometrics

logger = logging.getLogger(__name__)

# ── OSM filter registry ────────────────────────────────────────────────────────
OSM_FILTER_REGISTRY: dict[str, Any] = {
    "HEX2VEC_FILTER": HEX2VEC_FILTER,
    # ← Add more filters here
}


def get_osm_filter(name: str) -> Any:
    if name not in OSM_FILTER_REGISTRY:
        raise ValueError(
            f"Unknown OSM filter '{name}'. "
            f"Available: {list(OSM_FILTER_REGISTRY.keys())}"
        )
    return OSM_FILTER_REGISTRY[name]


def run_embedding_pipeline(
    embedder,
    embedder_name: str,
    regions_train: gpd.GeoDataFrame,
    regions_dev: gpd.GeoDataFrame,
    regions_test: gpd.GeoDataFrame,
    full_regions: gpd.GeoDataFrame,
    osm_filter: Any,
    neighbourhood_radius: int,
    fit_kwargs: dict,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Run the full embed pipeline for train / dev / test splits.

    Returns:
        (embeddings_train, embeddings_dev, embeddings_test) as GeoDataFrames
        with an added `h3` column holding the region index.
    """
    loader = OSMPbfLoader()
    joiner = IntersectionJoiner()

    # Buffer regions
    buf_train = ring_buffer_h3_regions_gdf(regions_train, neighbourhood_radius)
    buf_dev = ring_buffer_h3_regions_gdf(regions_dev, neighbourhood_radius)
    buf_test = ring_buffer_h3_regions_gdf(regions_test, neighbourhood_radius)

    # combined = gpd.GeoDataFrame(
    #     geometry=[
    #         unary_union(
    #             list(buf_train.geometry)
    #             + list(buf_dev.geometry)
    #             + list(buf_test.geometry)
    #         )
    #     ],
    #     crs=buf_train.crs,
    # )
    combined = gpd.GeoDataFrame(
        pd.concat([buf_train, buf_dev, buf_test], ignore_index=False), crs=buf_train.crs
    )

    combined = combined[~combined.index.duplicated(keep="first")]

    logger.info("Loading OSM features for combined region (single pass)...")

    # Compute Urban Morphometrics
    logger.info("Loading Urban Morphometrics features for all regions...")
    morpho_all = compute_urban_morphometrics(
        study_area_gdf=combined,
        pbf_path=None,
        run_name="king_county",
        output_folder="urban_morphometrics",
        neighbourhood_distance=300,
        num_quantiles=4,
        equal_area_crs="EPSG:5070",
        equidistant_crs="EPSG:2926",
        conformal_crs="EPSG:2926",
        n_workers=20,
        use_cache=True,
    )

    # Load OSM features
    logger.info("Loading OSM features for all regions...")
    osm_all = loader.load(combined, osm_filter)

    print(morpho_all.head())

    print(osm_all.head())

    # features_all = osm_all.join(morpho_all, how="outer", lsuffix="fc", rsuffix="um")
    #
    # print(features_all.head())

    logger.info("Joining OSM features for train regions...")
    joint_train = joiner.transform(buf_train, osm_all)

    logger.info("Joining OSM features for dev regions...")
    joint_dev = joiner.transform(buf_dev, osm_all)

    logger.info("Joining OSM features for test regions...")
    joint_test = joiner.transform(buf_test, osm_all)

    # Fit (if needed)
    if requires_fit(embedder_name):
        neighbourhood = H3Neighbourhood(buf_train)
        logger.info("Fitting embedder '%s'...", embedder_name)
        _fit_kwargs = dict(fit_kwargs)  # shallow copy
        if "trainer_kwargs" in _fit_kwargs:
            _fit_kwargs["trainer_kwargs"] = {
                **_fit_kwargs["trainer_kwargs"],
            }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedder.fit(
                regions_gdf=buf_train,
                features_gdf=osm_all,
                joint_gdf=joint_train,
                neighbourhood=neighbourhood,
                **_fit_kwargs,
            )
    else:
        logger.info("Embedder '%s' does not require fitting — skipping.", embedder_name)

    # Transform
    def _transform(buf, osm, joint, morpho, label):
        logger.info("Transforming %s split...", label)
        emb = embedder.transform(
            regions_gdf=buf,
            osm_features_gdf=osm,
            joint_gdf=joint,
            morpho_features_gdf=morpho,
        )
        emb["h3"] = emb.index
        return emb

    emb_train = _transform(buf_train, osm_all, joint_train, morpho_all, "train")
    emb_dev = _transform(buf_dev, osm_all, joint_dev, morpho_all, "dev")
    emb_test = _transform(buf_test, osm_all, joint_test, morpho_all, "test")

    return emb_train, emb_dev, emb_test
