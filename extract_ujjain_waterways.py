from pathlib import Path
from pyrosm import OSM

ROOT = Path(__file__).resolve().parent

PBF = ROOT / "data" / "raw" / "central-zone-latest.osm.pbf"

OUT = ROOT / "data" / "processed" / "ujjain" / "osm"
OUT.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUT / "ujjain_waterways.gpkg"

# Ujjain study-area bounding box
BBOX = [75.68, 23.12, 75.85, 23.25]

print("Loading regional OSM extract...")
osm = OSM(str(PBF), bounding_box=BBOX)

print("Extracting rivers, streams and canals...")

waterways = osm.get_data_by_custom_criteria(
    custom_filter={
        "waterway": ["river", "stream", "canal"]
    },
    filter_type="keep"
)

if waterways is None or waterways.empty:
    print("No waterways found in the study area.")
    raise SystemExit(1)

print("\nWaterway extraction successful.")
print("Total features:", len(waterways))

print("\nWaterway categories:")
print(waterways["waterway"].value_counts())

if "name" in waterways.columns:
    print("\nNamed waterways:")
    print(
        waterways.loc[
            waterways["name"].notna(),
            ["name", "waterway"]
        ].drop_duplicates().to_string(index=False)
    )

waterways.to_file(
    OUTPUT_FILE,
    layer="waterways",
    driver="GPKG"
)

print("\nSaved to:", OUTPUT_FILE)
print("CRS:", waterways.crs)
print("Geometry types:")
print(waterways.geometry.geom_type.value_counts())

