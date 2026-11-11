import geopandas as gpd

file = "data/processed/ujjain/osm/ujjain_road_edges.gpkg"

gdf = gpd.read_file(file)

print("===== ROAD DATASET =====")
print("Features:", len(gdf))
print("CRS:", gdf.crs)

print("\n===== COLUMNS =====")
print(gdf.columns.tolist())

print("\n===== SAMPLE RECORDS =====")
print(gdf.drop(columns="geometry").head(5).to_string())

print("\n===== GEOMETRY TYPES =====")
print(gdf.geometry.geom_type.value_counts())

print("\n===== SAMPLE GEOMETRY =====")
print(gdf.geometry.iloc[0])
