from pathlib import Path
import numpy as np

from src.evacuation.risk_mapper import build_guwahati_risk_map


def test_guwahati_risk_map_from_real_rasters():
    result = build_guwahati_risk_map(
        dem_path="data/raw/srtm_dem_study_area.tif",
        historical_event_paths=[
            "data/raw/DFO_4259_From_20150602_to_20150629.tif",
            "data/raw/DFO_4288_From_20150813_to_20150911.tif",
        ],
    )

    assert result.risk.ndim == 2
    assert result.risk.size > 0
    assert result.crs == "EPSG:32646"
    assert result.risk.shape == result.historical_flood.shape

    valid = result.risk[np.isfinite(result.risk)]

    assert valid.size > 0
    assert np.all(valid >= 0)
    assert np.all(valid <= 1)

    print("\nGuwahati risk map:")
    print("Shape:", result.shape)
    print("Valid pixels:", valid.size)
    print("Risk min:", float(valid.min()))
    print("Risk max:", float(valid.max()))
    print("Risk mean:", float(valid.mean()))
    print("Permanent water pixels:", int(result.permanent_water.sum()))
