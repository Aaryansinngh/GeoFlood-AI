import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from affine import Affine

from src.data.acquisition import ChirpsV3RegionalClient
from src.data.acquisition.base_client import AcquisitionStatus
from src.geospatial.region_manager import BoundingBox


@pytest.fixture
def bbox():
    return BoundingBox(
        min_lat=23.12,
        max_lat=23.25,
        min_lon=75.68,
        max_lon=75.85,
    )


@pytest.fixture
def client(tmp_path):
    return ChirpsV3RegionalClient(
        cache_dir=tmp_path / "cache",
        max_retries=1,
        minimum_coverage=0.8,
    )


@pytest.fixture
def transform():
    return Affine(
        0.05, 0, 75.65,
        0, -0.05, 23.25,
    )


def make_dates(start, count):
    first = datetime.strptime(start, "%Y-%m-%d")
    return [
        first + timedelta(days=i)
        for i in range(count)
    ]


def test_invalid_minimum_coverage(tmp_path):
    with pytest.raises(ValueError):
        ChirpsV3RegionalClient(
            cache_dir=tmp_path,
            minimum_coverage=0,
        )

    with pytest.raises(ValueError):
        ChirpsV3RegionalClient(
            cache_dir=tmp_path,
            minimum_coverage=1.1,
        )


def test_build_url(client):
    day = datetime(2024, 7, 15)

    url = client._build_url(day)

    assert url.endswith(
        "/2024/07/15/chirps-v3.0.rnl.2024.07.15.tif"
    )
    assert url.startswith(client.BASE_URL)


def test_daily_records_preserve_coordinates(
    client, transform
):
    dates = make_dates("2024-07-15", 1)

    arrays = {
        dates[0]: np.array(
            [[10.5, 20.5], [30.5, 40.5]],
            dtype=np.float32,
        )
    }

    records = client._generate_daily_records(
        dates, arrays, transform
    )

    assert len(records) == 4
    assert records[0]["rainfall_mm"] == 10.5
    assert records[0]["latitude"] == 23.225
    assert records[0]["longitude"] == 75.675

    assert records[-1]["rainfall_mm"] == 40.5


def test_daily_records_preserve_missing_values(
    client, transform
):
    dates = make_dates("2024-07-15", 1)

    arrays = {
        dates[0]: np.array(
            [[10.0, np.nan], [np.nan, 40.0]],
            dtype=np.float32,
        )
    }

    records = client._generate_daily_records(
        dates, arrays, transform
    )

    assert len(records) == 4
    assert records[0]["rainfall_mm"] == 10.0
    assert records[1]["rainfall_mm"] == ""
    assert records[2]["rainfall_mm"] == ""
    assert records[3]["rainfall_mm"] == 40.0


@pytest.mark.parametrize(
    "window_days,expected_records",
    [
        (3, 4),
        (7, 4),
        (30, 4),
    ],
)
def test_rolling_record_generation(
    client,
    transform,
    window_days,
    expected_records,
):
    dates = make_dates("2024-01-01", window_days)

    arrays = {
        day: np.full(
            (2, 2),
            2.0,
            dtype=np.float32,
        )
        for day in dates
    }

    records = client._generate_rolling_records(
        dates,
        arrays,
        transform,
        window_days,
    )

    assert len(records) == expected_records

    for record in records:
        assert record["rainfall_sum_mm"] == (
            2.0 * window_days
        )
        assert record["valid_days"] == window_days
        assert record["window_days"] == window_days
        assert record["coverage_fraction"] == 1.0


def test_rolling_window_rejects_insufficient_coverage(
    client,
    transform,
):
    dates = make_dates("2024-07-01", 3)

    arrays = {
        dates[0]: np.array(
            [[10.0, 10.0]],
            dtype=np.float32,
        ),
        dates[1]: np.array(
            [[20.0, np.nan]],
            dtype=np.float32,
        ),
        dates[2]: np.array(
            [[30.0, 30.0]],
            dtype=np.float32,
        ),
    }

    records = client._generate_rolling_records(
        dates,
        arrays,
        transform,
        3,
    )

    assert len(records) == 2

    # First pixel has 3 valid days.
    assert records[0]["rainfall_sum_mm"] == 60.0
    assert records[0]["valid_days"] == 3

    # Second pixel has only 2 valid days.
    # Required coverage is ceil(3 * 0.8) = 3.
    assert records[1]["rainfall_sum_mm"] == ""
    assert records[1]["valid_days"] == 2
    assert records[1]["coverage_fraction"] == pytest.approx(
        0.667, abs=0.001
    )


def test_rolling_window_seven_day_coverage(
    client,
    transform,
):
    dates = make_dates("2024-07-01", 7)

    arrays = {}

    for index, day in enumerate(dates):
        arrays[day] = np.array(
            [[1.0, 1.0 if index < 5 else np.nan]],
            dtype=np.float32,
        )

    records = client._generate_rolling_records(
        dates,
        arrays,
        transform,
        7,
    )

    # ceil(7 * 0.8) = 6 valid days required.
    assert records[0]["valid_days"] == 7
    assert records[0]["rainfall_sum_mm"] == 7.0

    assert records[1]["valid_days"] == 5
    assert records[1]["rainfall_sum_mm"] == ""


def test_cache_load_returns_cached_data(
    client,
    bbox,
):
    day = datetime(2024, 7, 15)
    cache_file = client._cache_path(
        "ujjain", bbox, day
    )

    cache_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected = np.array(
        [[10.0, 20.0], [30.0, 40.0]],
        dtype=np.float32,
    )

    transform = Affine(
        0.05, 0, 75.65,
        0, -0.05, 23.25,
    )

    np.savez_compressed(
        cache_file,
        rainfall=expected,
        transform=np.array([
            transform.a, transform.b, transform.c,
            transform.d, transform.e, transform.f,
        ]),
        row_off=10,
        col_off=20,
    )

    result = client._load_day(
        "ujjain", bbox, day
    )

    rainfall, actual_transform, row, col, cached = result

    np.testing.assert_array_equal(rainfall, expected)

    assert actual_transform == transform
    assert row == 10
    assert col == 20
    assert cached is True


def test_invalid_cache_is_reacquired(
    client,
    bbox,
    monkeypatch,
):
    day = datetime(2024, 7, 15)

    cache_file = client._cache_path(
        "ujjain", bbox, day
    )

    cache_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_file.write_text(
        "invalid cache content",
        encoding="utf-8",
    )

    expected = np.array(
        [[12.0, 14.0]],
        dtype=np.float32,
    )

    transform = Affine(
        0.05, 0, 75.65,
        0, -0.05, 23.25,
    )

    def fake_remote(url, requested_bbox):
        return expected, transform, 5, 10

    monkeypatch.setattr(
        client,
        "_read_remote_day",
        fake_remote,
    )

    result = client._load_day(
        "ujjain", bbox, day
    )

    rainfall, actual_transform, row, col, cached = result

    np.testing.assert_array_equal(rainfall, expected)

    assert actual_transform == transform
    assert row == 5
    assert col == 10
    assert cached is False
    assert cache_file.exists()


def test_write_csv_creates_valid_file(
    client,
    tmp_path,
):
    output = tmp_path / "rainfall.csv"

    records = [
        {
            "date": "2024-07-15",
            "rainfall_mm": 25.5,
        },
        {
            "date": "2024-07-16",
            "rainfall_mm": 30.5,
        },
    ]

    client._write_csv(
        output,
        ["date", "rainfall_mm"],
        records,
    )

    frame = pd.read_csv(output)

    assert len(frame) == 2
    assert list(frame.columns) == [
        "date", "rainfall_mm"
    ]
    assert frame.iloc[0]["rainfall_mm"] == 25.5


def test_acquire_creates_outputs_and_manifest(
    client,
    bbox,
    transform,
    tmp_path,
    monkeypatch,
):
    dates = make_dates("2024-07-15", 3)

    def fake_load_day(region_id, requested_bbox, day):
        index = (day - dates[0]).days

        rainfall = np.full(
            (2, 2),
            10.0 + index,
            dtype=np.float32,
        )

        return rainfall, transform, 0, 0, False

    monkeypatch.setattr(
        client,
        "_load_day",
        fake_load_day,
    )

    output_dir = tmp_path / "output"

    result = client.acquire(
        region_id="ujjain",
        bbox=bbox,
        start_date="2024-07-15",
        end_date="2024-07-17",
        output_dir=output_dir,
    )

    assert result.status == AcquisitionStatus.SUCCESS
    assert result.records_count == 3
    assert result.missing_count == 0

    expected_files = [
        "daily_rainfall.csv",
        "rainfall_3day.csv",
        "rainfall_7day.csv",
        "rainfall_30day.csv",
        "acquisition_manifest.json",
    ]

    for filename in expected_files:
        assert (output_dir / filename).exists()

    daily = pd.read_csv(
        output_dir / "daily_rainfall.csv"
    )

    rolling = pd.read_csv(
        output_dir / "rainfall_3day.csv"
    )

    assert len(daily) == 12
    assert len(rolling) == 4

    assert (rolling["valid_days"] == 3).all()
    assert np.allclose(
        rolling["rainfall_sum_mm"],
        33.0,
    )

    manifest = json.loads(
        (output_dir / "acquisition_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["region_id"] == "ujjain"
    assert manifest["requested_days"] == 3
    assert manifest["downloaded_days"] == 3
    assert manifest["missing_day_count"] == 0
    assert manifest["total_pixel_records"] == 12


def test_acquire_handles_missing_day(
    client,
    bbox,
    transform,
    tmp_path,
    monkeypatch,
):
    dates = make_dates("2024-07-15", 3)

    def fake_load_day(region_id, requested_bbox, day):
        if day == dates[1]:
            return None

        rainfall = np.full(
            (2, 2),
            10.0,
            dtype=np.float32,
        )

        return rainfall, transform, 0, 0, False

    monkeypatch.setattr(
        client,
        "_load_day",
        fake_load_day,
    )

    output_dir = tmp_path / "missing_output"

    result = client.acquire(
        region_id="ujjain",
        bbox=bbox,
        start_date="2024-07-15",
        end_date="2024-07-17",
        output_dir=output_dir,
    )

    assert result.status == AcquisitionStatus.SUCCESS
    assert result.records_count == 2
    assert result.missing_count == 1

    manifest = json.loads(
        (output_dir / "acquisition_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["missing_dates"] == [
        "2024-07-16"
    ]

    assert manifest["missing_pixel_count"] == 4
    assert manifest["missing_day_count"] == 1


def test_acquire_returns_unavailable_when_all_days_fail(
    client,
    bbox,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        client,
        "_load_day",
        lambda *args: None,
    )

    result = client.acquire(
        region_id="ujjain",
        bbox=bbox,
        start_date="2024-07-15",
        end_date="2024-07-16",
        output_dir=tmp_path / "unavailable",
    )

    assert result.status == AcquisitionStatus.UNAVAILABLE
    assert result.records_count == 0
    assert result.missing_count == 2
    assert result.local_path is None


def test_acquire_rejects_invalid_date_range(
    client,
    bbox,
    tmp_path,
):
    result = client.acquire(
        region_id="ujjain",
        bbox=bbox,
        start_date="2024-07-20",
        end_date="2024-07-15",
        output_dir=tmp_path / "invalid",
    )

    assert result.status == AcquisitionStatus.INVALID_INPUT
    assert result.local_path is None
