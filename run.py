from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf

from urban_morphometrics.main import compute_urban_morphometrics

def run_main():

    pbf_url = "https://download.geofabrik.de/europe/germany/baden-wuerttemberg/karlsruhe-regbez-latest.osm.pbf"
    region_gdf = geocode_to_region_gdf("Südstadt, Karlsruhe, Germany")
    regionalizer = H3Regionalizer(resolution=9)
    regions_gdf = regionalizer.transform(region_gdf)
    print(f"Number of cells: {len(regions_gdf)}")

    compute_urban_morphometrics(
        study_area_gdf=regions_gdf,
        pbf_path=pbf_url,
        run_name="my_run",
        neighbourhood_distance=200,
        num_quantiles=4,
        # metrics=["volume", "floor_area"],
        debug=True,
        output_folder="/mnt/c/Users/Isaac/Downloads/urban_morphometrics"
    )

if __name__ == "__main__":
    run_main()