import geopandas as gpd
from shapely.geometry import Point

file = "data/processed/ujjain/osm/ujjain_waterways.gpkg"

gdf = gpd.read_file(file)

print("\n===== WATERWAY ATTRIBUTES =====")
print("Columns:", list(gdf.columns))

print("\n===== ALL FIVE RIVER FEATURES =====")

metric = gdf.to_crs("EPSG:32643")

ram_ghat = gpd.GeoSeries(
    [Point(75.7683, 23.1828)],
    crs="EPSG:4326"
).to_crs("EPSG:32643").iloc[0]

for index, row in gdf.iterrows():
    geometry = metric.loc[index].geometry if hasattr(metric.loc[index], "geometry") else metric.loc[index]

    print(f"\nFeature {index + 1}")

    for col in gdf.columns:
        if col != "geometry":
            print(f"{col}: {row[col]}")

    print(f"Length: {geometry.length:.1f} metres")
    print(f"Distance from Ram Ghat: {geometry.distance(ram_ghat):.1f} metres")

print("\n===== COMPLETE =====")
