from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf

from urban_morphometrics.main import compute_urban_morphometrics

def run_main():

    pbf_url = "https://download.geofabrik.de/europe/germany/baden-wuerttemberg/karlsruhe-regbez-latest.osm.pbf"
    region_gdf = geocode_to_region_gdf("Südstadt, Karlsruhe, Germany")
    regionalizer = H3Regionalizer(resolution=9)
    regions_gdf = regionalizer.transform(region_gdf)
    print(f"Number of cells: {len(regions_gdf)}")
    
    metric_config = {
        "knn_k": 15,
        "tessellation_buffer": 5,
        "tessellation_min_buffer": 0,
        "tessellation_max_buffer": 10,
        "tessellation_shrink": 0.4,
        "tessellation_segment": 0.5,
        "street_alignment_max_distance": 500.0
    }

    compute_urban_morphometrics(
        study_area_gdf=regions_gdf,
        pbf_path=pbf_url,
        run_name="my_run3",
        neighbourhood_distance=200,
        num_quantiles=4,
        metric_config=metric_config,
        metrics=["courtyards"],
        debug=True,
        output_folder="/mnt/c/Users/Isaac/Downloads/urban_morphometrics",
        use_cache=False
    )

if __name__ == "__main__":
    run_main()