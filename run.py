from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf

from urban_morphometrics.main import compute_urban_morphometrics

pbf_url = "https://download.geofabrik.de/europe/germany/baden-wuerttemberg/karlsruhe-regbez-latest.osm.pbf"
region_gdf = geocode_to_region_gdf("Südstadt, Karlsruhe, Germany")
regionalizer = H3Regionalizer(resolution=9)
regions_gdf = regionalizer.transform(region_gdf)
print(f"Number of cells: {len(regions_gdf)}")

compute_urban_morphometrics(
    study_area_gdf=regions_gdf,
    pbf_path=pbf_url,
    run_name="my_run",
    output_folder="./output",
    metrics=["volume", "floor_area"],
    debug=True,
)