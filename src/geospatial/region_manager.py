"""
src/geospatial/region_manager.py

GeoFlood-AI 2.0 Regional Configuration & Geospatial Boundary Manager.
Replaces single-basin hardcoded logic with a modular multi-region manager supporting:
1. Assam (Brahmaputra Floodplain)
2. Kerala (Periyar-Pamba Catchments)
3. Mahanadi (Odisha Delta)
4. Godavari (Lower Basin & Delta)

Provides:
- YAML regional configuration loading with caching
- Coordinate validation & bounding-box / polygon containment checks
- CRS transformations (WGS84 EPSG:4326 <-> Regional UTM metric projections)
- Dynamic master grid dimensions & affine transform computation
- SRTM tile identification for arbitrary bounding boxes
- Backward compatibility with legacy config/settings.yaml
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import math
import logging
import yaml
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def __post_init__(self):
        if self.min_lat >= self.max_lat:
            raise ValueError(f"Invalid latitude bounds: min_lat ({self.min_lat}) >= max_lat ({self.max_lat})")
        if self.min_lon >= self.max_lon:
            raise ValueError(f"Invalid longitude bounds: min_lon ({self.min_lon}) >= max_lon ({self.max_lon})")
        if not (-90.0 <= self.min_lat <= 90.0 and -90.0 <= self.max_lat <= 90.0):
            raise ValueError(f"Latitudes out of [-90, 90]: ({self.min_lat}, {self.max_lat})")
        if not (-180.0 <= self.min_lon <= 180.0 and -180.0 <= self.max_lon <= 180.0):
            raise ValueError(f"Longitudes out of [-180, 180]: ({self.min_lon}, {self.max_lon})")

    def contains(self, lat: float, lon: float) -> bool:
        return (self.min_lat <= lat <= self.max_lat) and (self.min_lon <= lon <= self.max_lon)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Returns (min_lon, min_lat, max_lon, max_lat)."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    def as_dict(self) -> Dict[str, float]:
        return {
            "min_lat": self.min_lat,
            "max_lat": self.max_lat,
            "min_lon": self.min_lon,
            "max_lon": self.max_lon,
        }


@dataclass
class RegionMetadata:
    id: str
    name: str
    description: str
    river_basin: str
    physiographic_zone: str
    bbox: BoundingBox
    centroid: Tuple[float, float]
    target_crs: str
    projected_metric_crs: str
    target_resolution_deg: float
    target_resolution_meters: float
    events: Dict[str, Any]
    terrain_cfg: Dict[str, Any]
    landmarks: List[Dict[str, Any]] = field(default_factory=list)
    vulnerable_districts: List[str] = field(default_factory=list)
    polygon_coords: Optional[List[Tuple[float, float]]] = None


class RegionManager:
    """
    Central registry and manager for multi-regional flood analysis.
    Supports loading from configs/regions or config/regions.
    """

    SUPPORTED_REGIONS = ["assam", "kerala", "mahanadi", "godavari"]

    def __init__(self, config_dirs: Optional[List[Union[str, Path]]] = None):
        if config_dirs is None:
            self.config_dirs = [
                Path("configs/regions"),
                Path("config/regions"),
                Path("config"),
            ]
        else:
            self.config_dirs = [Path(p) for p in config_dirs]

        self._regions: Dict[str, RegionMetadata] = {}
        self._load_all_regions()

    def _find_region_file(self, region_id: str) -> Optional[Path]:
        for cdir in self.config_dirs:
            p = cdir / f"{region_id}.yaml"
            if p.exists():
                return p
            p_yml = cdir / f"{region_id}.yml"
            if p_yml.exists():
                return p_yml
        return None

    def _load_all_regions(self):
        for reg_id in self.SUPPORTED_REGIONS:
            fpath = self._find_region_file(reg_id)
            if fpath and fpath.exists():
                try:
                    self.load_region(reg_id, fpath)
                except Exception as e:
                    logger.warning("Failed loading region config for '%s' from %s: %s", reg_id, fpath, e)

        # Legacy backward-compatibility fallback
        if "assam" not in self._regions:
            legacy_settings = Path("config/settings.yaml")
            if legacy_settings.exists():
                self._load_legacy_settings(legacy_settings)

    def _load_legacy_settings(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        reg_cfg = cfg.get("region", {})
        b = reg_cfg.get("bbox", {})
        bbox = BoundingBox(
            min_lat=float(b.get("min_lat", 26.30)),
            max_lat=float(b.get("max_lat", 27.20)),
            min_lon=float(b.get("min_lon", 92.80)),
            max_lon=float(b.get("max_lon", 94.20)),
        )
        meta = RegionMetadata(
            id="assam",
            name=reg_cfg.get("name", "Assam_Brahmaputra"),
            description=reg_cfg.get("description", "Brahmaputra Floodplain"),
            river_basin="Brahmaputra",
            physiographic_zone="Himalayan Foreland Alluvial Floodplain",
            bbox=bbox,
            centroid=((bbox.min_lat + bbox.max_lat) / 2.0, (bbox.min_lon + bbox.max_lon) / 2.0),
            target_crs=reg_cfg.get("target_crs", "EPSG:4326"),
            projected_metric_crs="EPSG:32646",
            target_resolution_deg=float(reg_cfg.get("target_resolution_deg", 0.002245788)),
            target_resolution_meters=250.0,
            events=cfg.get("events", {}),
            terrain_cfg=cfg.get("terrain", {}),
            landmarks=[],
            vulnerable_districts=[],
        )
        self._regions["assam"] = meta

    def load_region(self, region_id: str, file_path: Path) -> RegionMetadata:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        r_data = raw.get("region", {})
        b_data = r_data.get("bbox", {})
        bbox = BoundingBox(
            min_lat=float(b_data["min_lat"]),
            max_lat=float(b_data["max_lat"]),
            min_lon=float(b_data["min_lon"]),
            max_lon=float(b_data["max_lon"]),
        )
        cent = r_data.get("centroid", {})
        centroid = (
            float(cent.get("lat", (bbox.min_lat + bbox.max_lat) / 2.0)),
            float(cent.get("lon", (bbox.min_lon + bbox.max_lon) / 2.0)),
        )

        meta = RegionMetadata(
            id=region_id,
            name=r_data.get("name", region_id),
            description=r_data.get("description", ""),
            river_basin=r_data.get("river_basin", "Unknown"),
            physiographic_zone=r_data.get("physiographic_zone", "Unknown"),
            bbox=bbox,
            centroid=centroid,
            target_crs=r_data.get("target_crs", "EPSG:4326"),
            projected_metric_crs=r_data.get("projected_metric_crs", "EPSG:3857"),
            target_resolution_deg=float(r_data.get("target_resolution_deg", 0.002245788)),
            target_resolution_meters=float(r_data.get("target_resolution_meters", 250.0)),
            events=raw.get("events", {}),
            terrain_cfg=raw.get("terrain", {}),
            landmarks=raw.get("landmarks", []),
            vulnerable_districts=raw.get("vulnerable_districts", []),
            polygon_coords=r_data.get("polygon_coords"),
        )
        self._regions[region_id] = meta
        return meta

    def get_region(self, region_id: str) -> RegionMetadata:
        reg_norm = region_id.lower().strip()
        if reg_norm not in self._regions:
            available = list(self._regions.keys())
            raise KeyError(f"Region '{region_id}' not found. Available regions: {available}")
        return self._regions[reg_norm]

    def list_regions(self) -> List[str]:
        return list(self._regions.keys())

    def validate_coordinates(self, region_id: str, lat: float, lon: float) -> Tuple[bool, Optional[str]]:
        """Validate if coordinates are numeric, non-NaN, and fall within region bounding box."""
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except (ValueError, TypeError):
            return False, f"Coordinates must be numeric, received lat={lat}, lon={lon}"

        if math.isnan(lat_f) or math.isnan(lon_f) or math.isinf(lat_f) or math.isinf(lon_f):
            return False, "Coordinates cannot be NaN or Infinite"

        region = self.get_region(region_id)
        if not region.bbox.contains(lat_f, lon_f):
            return (
                False,
                f"Coordinates ({lat_f:.4f}, {lon_f:.4f}) fall outside '{region.name}' bounding box: "
                f"lat [{region.bbox.min_lat:.2f}, {region.bbox.max_lat:.2f}], "
                f"lon [{region.bbox.min_lon:.2f}, {region.bbox.max_lon:.2f}]",
            )

        if region.polygon_coords:
            # Ray casting point-in-polygon check
            if not self._point_in_polygon(lat_f, lon_f, region.polygon_coords):
                return False, f"Coordinates ({lat_f:.4f}, {lon_f:.4f}) fall outside region boundary polygon"

        return True, None

    @staticmethod
    def _point_in_polygon(lat: float, lon: float, polygon: List[Tuple[float, float]]) -> bool:
        """Ray-casting algorithm for point-in-polygon test."""
        n = len(polygon)
        inside = False
        p1_lat, p1_lon = polygon[0]
        for i in range(1, n + 1):
            p2_lat, p2_lon = polygon[i % n]
            if lon > min(p1_lon, p2_lon):
                if lon <= max(p1_lon, p2_lon):
                    if lat <= max(p1_lat, p2_lat):
                        if p1_lon != p2_lon:
                            xinters = (lon - p1_lon) * (p2_lat - p1_lat) / (p2_lon - p1_lon) + p1_lat
                        if p1_lat == p2_lat or lat <= xinters:
                            inside = not inside
            p1_lat, p1_lon = p2_lat, p2_lon
        return inside

    def compute_grid_dimensions(self, region_id: str) -> Tuple[int, int, Tuple[float, float, float, float, float, float]]:
        """
        Computes the discrete raster grid shape (height, width) and affine transform tuple
        (c, a, b, f, d, e) for the given region.
        """
        region = self.get_region(region_id)
        res_deg = region.target_resolution_deg
        bbox = region.bbox

        height = int(math.ceil((bbox.max_lat - bbox.min_lat) / res_deg))
        width = int(math.ceil((bbox.max_lon - bbox.min_lon) / res_deg))

        # Affine transform: x_min, res_x, 0, y_max, 0, -res_y
        transform_tuple = (bbox.min_lon, res_deg, 0.0, bbox.max_lat, 0.0, -res_deg)
        return height, width, transform_tuple

    @staticmethod
    def get_srtm_tiles_for_bbox(bbox: BoundingBox) -> List[str]:
        """
        Determine standard 1x1 degree SRTMGL1 tile filenames covering any arbitrary bounding box.
        Format: NxxEyyy (e.g. N26E092)
        """
        lat_start = int(math.floor(bbox.min_lat))
        lat_end = int(math.floor(bbox.max_lat))
        lon_start = int(math.floor(bbox.min_lon))
        lon_end = int(math.floor(bbox.max_lon))

        tiles = []
        for lat in range(lat_start, lat_end + 1):
            for lon in range(lon_start, lon_end + 1):
                ns = "N" if lat >= 0 else "S"
                ew = "E" if lon >= 0 else "W"
                tile_name = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"
                tiles.append(tile_name)
        return sorted(list(set(tiles)))

    def transform_point_crs(self, lat: float, lon: float, src_crs: str, dst_crs: str) -> Tuple[float, float]:
        """
        Transform a point between coordinate systems (e.g. WGS84 EPSG:4326 to UTM).
        Uses pyproj if available, otherwise falls back to equirectangular / UTM formula approximation.
        """
        if src_crs == dst_crs:
            return lat, lon

        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            # pyproj expects (x, y) = (lon, lat) in geographic
            x_dst, y_dst = transformer.transform(lon, lat)
            return y_dst, x_dst
        except ImportError:
            # Fallback simple metric approximation if pyproj is absent
            if dst_crs.startswith("EPSG:326"):
                # Approximate UTM metric coordinates from WGS84
                dx = (lon - 0.0) * 111320.0 * math.cos(math.radians(lat))
                dy = lat * 110540.0
                return dy, dx
            return lat, lon
