"""
src/data/validation/leakage_guard.py

Temporal & Feature Leakage Prevention Engine for GeoFlood-AI 2.0.
Explicitly distinguishes between:
1. Pre-event features (available prior to flood onset, valid for operational forecasting)
2. Concurrent / during-event features (accumulating during flood duration)
3. Post-event features (available only after the flood event window closes, valid strictly for retrospective mapping)

Provides:
- Configurable prediction cutoff (lead time prior to event onset)
- Automated feature inspection and rejection for forecasting mode
- Validation report generation
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelingMode(str, Enum):
    RETROSPECTIVE_SUSCEPTIBILITY = "retrospective_susceptibility"
    PRE_EVENT_FORECASTING = "pre_event_forecasting"
    EVENT_BASED_ANALYSIS = "event_based_analysis"


class FeatureTemporalType(str, Enum):
    STATIC_TERRAIN = "static_terrain"
    PRE_EVENT_HYDROMET = "pre_event_hydromet"
    DURING_EVENT_HYDROMET = "during_event_hydromet"
    POST_EVENT_COMPOSITE = "post_event_composite"


@dataclass
class FeatureDefinition:
    name: str
    temporal_type: FeatureTemporalType
    description: str
    relative_window_days: Tuple[int, int]  # (start_day_offset, end_day_offset) relative to onset (day 0)
    allowed_modes: List[ModelingMode]


# Standard feature catalog classification
FEATURE_REGISTRY: Dict[str, FeatureDefinition] = {
    "elevation": FeatureDefinition(
        name="elevation",
        temporal_type=FeatureTemporalType.STATIC_TERRAIN,
        description="SRTM / Copernicus Digital Elevation Model (m)",
        relative_window_days=(-9999, -9999),
        allowed_modes=[
            ModelingMode.RETROSPECTIVE_SUSCEPTIBILITY,
            ModelingMode.PRE_EVENT_FORECASTING,
            ModelingMode.EVENT_BASED_ANALYSIS,
        ],
    ),
    "slope": FeatureDefinition(
        name="slope",
        temporal_type=FeatureTemporalType.STATIC_TERRAIN,
        description="Terrain Slope (degrees)",
        relative_window_days=(-9999, -9999),
        allowed_modes=[
            ModelingMode.RETROSPECTIVE_SUSCEPTIBILITY,
            ModelingMode.PRE_EVENT_FORECASTING,
            ModelingMode.EVENT_BASED_ANALYSIS,
        ],
    ),
    "tpi": FeatureDefinition(
        name="tpi",
        temporal_type=FeatureTemporalType.STATIC_TERRAIN,
        description="Topographic Position Index (m)",
        relative_window_days=(-9999, -9999),
        allowed_modes=[
            ModelingMode.RETROSPECTIVE_SUSCEPTIBILITY,
            ModelingMode.PRE_EVENT_FORECASTING,
            ModelingMode.EVENT_BASED_ANALYSIS,
        ],
    ),
    "relative_elevation": FeatureDefinition(
        name="relative_elevation",
        temporal_type=FeatureTemporalType.STATIC_TERRAIN,
        description="Relative height above local minimum / HAND proxy (m)",
        relative_window_days=(-9999, -9999),
        allowed_modes=[
            ModelingMode.RETROSPECTIVE_SUSCEPTIBILITY,
            ModelingMode.PRE_EVENT_FORECASTING,
            ModelingMode.EVENT_BASED_ANALYSIS,
        ],
    ),
    "rain_3d": FeatureDefinition(
        name="rain_3d",
        temporal_type=FeatureTemporalType.PRE_EVENT_HYDROMET,
        description="3-day antecedent rainfall prior to flood onset (mm)",
        relative_window_days=(-3, 0),
        allowed_modes=[
            ModelingMode.RETROSPECTIVE_SUSCEPTIBILITY,
            ModelingMode.PRE_EVENT_FORECASTING,
            ModelingMode.EVENT_BASED_ANALYSIS,
        ],
    ),
    "rain_7d": FeatureDefinition(
        name="rain_7d",
        temporal_type=FeatureTemporalType.PRE_EVENT_HYDROMET,
        description="7-day antecedent rainfall prior to flood onset (mm)",
        relative_window_days=(-7, 0),
        allowed_modes=[
            ModelingMode.RETROSPECTIVE_SUSCEPTIBILITY,
            ModelingMode.PRE_EVENT_FORECASTING,
            ModelingMode.EVENT_BASED_ANALYSIS,
        ],
    ),
    "rain_14d": FeatureDefinition(
        name="rain_14d",
        temporal_type=FeatureTemporalType.PRE_EVENT_HYDROMET,
        description="14-day antecedent watershed saturation rainfall (mm)",
        relative_window_days=(-14, 0),
        allowed_modes=[
            ModelingMode.RETROSPECTIVE_SUSCEPTIBILITY,
            ModelingMode.PRE_EVENT_FORECASTING,
            ModelingMode.EVENT_BASED_ANALYSIS,
        ],
    ),
    "rain_event_sum": FeatureDefinition(
        name="rain_event_sum",
        temporal_type=FeatureTemporalType.POST_EVENT_COMPOSITE,
        description="Cumulative precipitation across entire event window (includes post-onset rainfall)",
        relative_window_days=(0, 28),
        allowed_modes=[
            ModelingMode.RETROSPECTIVE_SUSCEPTIBILITY,
            ModelingMode.EVENT_BASED_ANALYSIS,
        ],
    ),
}


class LeakageGuard:
    """
    Enforces scientific integrity on feature sets according to the selected modeling mode.
    """

    def __init__(self, mode: ModelingMode = ModelingMode.RETROSPECTIVE_SUSCEPTIBILITY, lead_time_hours: float = 0.0):
        self.mode = mode
        self.lead_time_hours = lead_time_hours

    def validate_features(self, feature_names: List[str]) -> Tuple[bool, List[str]]:
        """
        Validates feature list against the current modeling mode.
        Returns (is_valid, list_of_violations).
        """
        violations = []
        for feat in feature_names:
            if feat not in FEATURE_REGISTRY:
                logger.warning("Unregistered feature '%s' encountered in leakage audit", feat)
                continue

            definition = FEATURE_REGISTRY[feat]
            if self.mode not in definition.allowed_modes:
                violations.append(
                    f"Feature '{feat}' has temporal type '{definition.temporal_type.value}' "
                    f"which is prohibited in mode '{self.mode.value}' (allowed only in: "
                    f"{[m.value for m in definition.allowed_modes]})."
                )

        is_valid = len(violations) == 0
        return is_valid, violations

    def filter_allowed_features(self, feature_names: List[str]) -> List[str]:
        """Returns only the features permissible under current mode."""
        allowed = []
        for feat in feature_names:
            if feat in FEATURE_REGISTRY:
                if self.mode in FEATURE_REGISTRY[feat].allowed_modes:
                    allowed.append(feat)
            else:
                # Default allow unregistered static features
                allowed.append(feat)
        return allowed

    def get_event_temporal_windows(
        self, event_began: str, event_ended: str
    ) -> Dict[str, Dict[str, str]]:
        """
        Compute explicit date intervals for pre-event, event-window, and lead-time cutoffs.
        """
        onset_dt = datetime.strptime(event_began, "%Y-%m-%d")
        end_dt = datetime.strptime(event_ended, "%Y-%m-%d")
        cutoff_dt = onset_dt - timedelta(hours=self.lead_time_hours)

        return {
            "prediction_cutoff": cutoff_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "pre_event_3d_window": {
                "start": (cutoff_dt - timedelta(days=3)).strftime("%Y-%m-%d"),
                "end": cutoff_dt.strftime("%Y-%m-%d"),
            },
            "pre_event_7d_window": {
                "start": (cutoff_dt - timedelta(days=7)).strftime("%Y-%m-%d"),
                "end": cutoff_dt.strftime("%Y-%m-%d"),
            },
            "pre_event_14d_window": {
                "start": (cutoff_dt - timedelta(days=14)).strftime("%Y-%m-%d"),
                "end": cutoff_dt.strftime("%Y-%m-%d"),
            },
            "event_active_window": {
                "start": onset_dt.strftime("%Y-%m-%d"),
                "end": end_dt.strftime("%Y-%m-%d"),
            },
        }
