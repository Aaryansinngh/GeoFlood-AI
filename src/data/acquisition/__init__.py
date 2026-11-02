"""
src/data/acquisition package for GeoFlood-AI 2.0.
"""

from src.data.acquisition.base_client import (
    BaseAcquisitionClient,
    AcquisitionResult,
    AcquisitionStatus,
)
from src.data.acquisition.nasa_power import NasaPowerClient
from src.data.acquisition.chirps import ChirpsClient
from src.data.acquisition.era5 import Era5Client
from src.data.acquisition.dem_client import DemClient
from src.data.acquisition.gfd_client import GfdClient

__all__ = [
    "BaseAcquisitionClient",
    "AcquisitionResult",
    "AcquisitionStatus",
    "NasaPowerClient",
    "ChirpsClient",
    "Era5Client",
    "DemClient",
    "GfdClient",
]
