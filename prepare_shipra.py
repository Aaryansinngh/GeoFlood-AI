import geopandas as gpd
import json

source = "data/processed/ujjain/osm/ujjain_waterways.gpkg"
output = "data/processed/ujjain/osm/ujjain_shipra_river.geojson"

gdf = gpd.read_file(source)

shipra = gdf[
    gdf["tags"].fillna("").str.contains(
        '"name":"Shipra"',
        case=False
    )
].copy()

if shipra.empty:
    raise RuntimeError("Shipra geometry not found.")

shipra = shipra.to_crs("EPSG:4326")
shipra.to_file(output, driver="GeoJSON")

print("Shipra geometry exported successfully.")
print("Features:", len(shipra))
print("Length (km):", round(
    shipra.to_crs("EPSG:32643").geometry.length.sum() / 1000, 2
))
print("Output:", output)
