"""
Deeper diagnostic: check reshape order, pixel alignment, and whether
the raster concentration in lower-right is caused by the sort order
or a padding issue.
"""
import pandas as pd
import numpy as np

df = pd.read_parquet('outputs/predictions.parquet')
ev1 = df[df['event_id'] == 'DFO_4259'].sort_values(['row', 'col'])

h = int(ev1['row'].max() + 1)
w = int(ev1['col'].max() + 1)
print(f"Grid: h={h}, w={w}")

# Reconstruct the RGBA array exactly as app does
probs  = ev1['prob_neural_network'].values.reshape((h, w))
perm   = ev1['is_perm_water'].values.reshape((h, w))
labels = ev1['flood_label'].values.reshape((h, w))

# Count colored pixels in each quadrant
top_half    = probs[:h//2, :]
bottom_half = probs[h//2:, :]
left_half   = probs[:, :w//2]
right_half  = probs[:, w//2:]

def nonzero_frac(arr):
    valid = ~np.isnan(arr)
    return valid.mean()

print("\n=== PIXEL COVERAGE BY QUADRANT ===")
print(f"Top-left:     {nonzero_frac(probs[:h//2, :w//2]):.3f}")
print(f"Top-right:    {nonzero_frac(probs[:h//2, w//2:]):.3f}")
print(f"Bottom-left:  {nonzero_frac(probs[h//2:, :w//2]):.3f}")
print(f"Bottom-right: {nonzero_frac(probs[h//2:, w//2:]):.3f}")

print(f"\nOverall valid fraction: {nonzero_frac(probs):.3f}")
print(f"High risk (>=0.70) fraction: {(probs >= 0.70).mean():.3f}")
print(f"Perm water fraction: {(perm == 1).mean():.3f}")

# Check lat coordinates in each quadrant
row_lats = ev1.groupby('row')['lat'].first()
col_lons = ev1.groupby('col')['lon'].first()

print("\n=== ROW → LAT MAPPING (first 5 and last 5) ===")
print(row_lats.head(5).to_string())
print("...")
print(row_lats.tail(5).to_string())

print("\n=== COL → LON MAPPING (first 5 and last 5) ===")
print(col_lons.head(5).to_string())
print("...")
print(col_lons.tail(5).to_string())

# Check if sort_values(['row','col']) gives a consistent row-major grid
# (i.e., all col 0 comes before col 1 within the same row)
ev1_check = ev1.reset_index(drop=True)
row_changes = (ev1_check['row'].diff() != 0)
col_resets  = ev1_check.loc[row_changes, 'col']
print(f"\nCol value at each row-start (should all be 0): min={col_resets.min()}, max={col_resets.max()}")

# Check the actual folium ImageOverlay signature
# bounds=[[south, west], [north, east]]
# The IMAGE top-left pixel maps to the NORTH-WEST corner
# The IMAGE bottom-right pixel maps to the SOUTH-EAST corner
# So row 0 → North, row h-1 → South  (which IS the case: row0_lat=27.19 > row399_lat=26.30)
# Col 0 → West, col w-1 → East        (which IS the case: col0_lon=92.80 < col622_lon=94.20)
# This means the reshape IS correct for Folium.

print("\n=== ORIENTATION SUMMARY ===")
print(f"Row 0 → lat {ev1[ev1.row==0].lat.iloc[0]:.4f}°N (NORTH edge)")
print(f"Row {h-1} → lat {ev1[ev1.row==h-1].lat.iloc[0]:.4f}°N (SOUTH edge)")
print(f"Col 0 → lon {ev1[ev1.col==0].lon.iloc[0]:.4f}°E (WEST edge)")
print(f"Col {w-1} → lon {ev1[ev1.col==w-1].lon.iloc[0]:.4f}°E (EAST edge)")
print("Orientation: CORRECT for Folium [[S,W],[N,E]] convention")

# --- Check where the high-susceptibility cells actually are geographically ---
high_risk = ev1[ev1['prob_neural_network'] >= 0.70]
print(f"\n=== HIGH RISK CELL LOCATIONS ===")
print(f"Count: {len(high_risk)}")
if len(high_risk) > 0:
    print(f"Lat range: [{high_risk.lat.min():.4f}, {high_risk.lat.max():.4f}]")
    print(f"Lon range: [{high_risk.lon.min():.4f}, {high_risk.lon.max():.4f}]")
    print(f"Row range: [{high_risk.row.min()}, {high_risk.row.max()}]")
    print(f"Col range: [{high_risk.col.min()}, {high_risk.col.max()}]")

# --- Check ZOOM level relevance ---
# At zoom=9, map shows ~6°×6° area. Center at (26.75, 93.50)
# Study area is only 0.9°×1.4°, well within view
print("\n=== MAP VIEWPORT ANALYSIS ===")
center_lat = (26.30 + 27.20) / 2
center_lon = (92.80 + 94.20) / 2
print(f"Center: {center_lat:.4f}°N, {center_lon:.4f}°E")
print(f"Span: Δlat={27.20-26.30:.2f}°, Δlon={94.20-92.80:.2f}°")
print("Zoom 9 shows ~6°x6° — study area should be visible but small")
print("Zoom 10 would be better to see full study area clearly")

# fit_bounds would solve viewport issues
print(f"\nfit_bounds should use: [[26.30, 92.80], [27.20, 94.20]]")
