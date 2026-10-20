import pandas as pd
import numpy as np

df = pd.read_parquet('outputs/predictions.parquet')
ev1 = df[df['event_id'] == 'DFO_4259'].sort_values(['row', 'col'])

actual_min_lat = ev1.lat.min()
actual_max_lat = ev1.lat.max()
actual_min_lon = ev1.lon.min()
actual_max_lon = ev1.lon.max()

pixel_size = 0.00224579
true_west  = actual_min_lon - pixel_size / 2
true_east  = actual_max_lon + pixel_size / 2
true_south = actual_min_lat - pixel_size / 2
true_north = actual_max_lat + pixel_size / 2

print("=== CONFIG BBOX (what app uses) ===")
print("SW=[26.30, 92.80]  NE=[27.20, 94.20]")

print("\n=== ACTUAL DATA BOUNDS ===")
print(f"Cell-center lat: [{actual_min_lat:.6f}, {actual_max_lat:.6f}]")
print(f"Cell-center lon: [{actual_min_lon:.6f}, {actual_max_lon:.6f}]")
print(f"True edge bounds: W={true_west:.6f} S={true_south:.6f} E={true_east:.6f} N={true_north:.6f}")

print("\n=== RASTER ORIENTATION ===")
row0_lat   = ev1[ev1.row==0].lat.iloc[0]
row399_lat = ev1[ev1.row==399].lat.iloc[0]
col0_lon   = ev1[ev1.col==0].lon.iloc[0]
col622_lon = ev1[ev1.col==622].lon.iloc[0]
print(f"Row 0 lat   = {row0_lat:.6f}  (should be NORTH ~{actual_max_lat:.4f})")
print(f"Row 399 lat = {row399_lat:.6f}  (should be SOUTH ~{actual_min_lat:.4f})")
print(f"Col 0 lon   = {col0_lon:.6f}  (should be WEST  ~{actual_min_lon:.4f})")
print(f"Col 622 lon = {col622_lon:.6f}  (should be EAST  ~{actual_max_lon:.4f})")
print(f"Row0 = NORTH? {abs(row0_lat - actual_max_lat) < 0.01}")

print("\n=== GRID COMPLETENESS ===")
h = int(ev1['row'].max() + 1)
w = int(ev1['col'].max() + 1)
print(f"Grid h={h}, w={w}, total={h*w}, actual rows={len(ev1)}, match={len(ev1)==h*w}")
dupes = ev1.duplicated(subset=['row','col']).sum()
print(f"Duplicate (row,col) pairs: {dupes}")

print("\n=== RESHAPE TEST ===")
prob_arr = ev1['prob_neural_network'].values.reshape((h, w))
print(f"Reshaped array shape: {prob_arr.shape}")
print(f"Top row (row 0) NaN fraction: {np.isnan(prob_arr[0]).mean():.3f}")
print(f"Bottom row (row 399) NaN fraction: {np.isnan(prob_arr[399]).mean():.3f}")

print("\n=== FOLIUM BOUNDS CHECK ===")
print("Folium ImageOverlay expects:  bounds=[[south, west], [north, east]]")
print("Row 0 = NORTH = image TOP  -- Folium renders image TOP at NORTH edge")
print(f"So correct bounds are: [[{true_south:.4f}, {true_west:.4f}], [{true_north:.4f}, {true_east:.4f}]]")
print(f"App is using:           [[26.3000, 92.8000], [27.2000, 94.2000]]")
print(f"Difference S: {26.30 - true_south:.6f} deg")
print(f"Difference N: {27.20 - true_north:.6f} deg")
print(f"Difference W: {92.80 - true_west:.6f} deg")
print(f"Difference E: {94.20 - true_east:.6f} deg")
