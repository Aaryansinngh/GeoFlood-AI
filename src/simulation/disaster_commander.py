"""
src/simulation/disaster_commander.py

Scenario-Based Flood Commander Engine
=====================================
A professional disaster management command simulation module built on real
historical flood events (DFO_4259 June 2015 and DFO_4288 August 2015, Assam).

Features:
- Realistic phased disaster timeline:
  Phase 1: Pre-Monsoon Early Watch (t - 48h)
  Phase 2: Heavy Inundation Onset (t 0h)
  Phase 3: Peak River Discharge & Breaches (t + 24h)
  Phase 4: Post-Flood Stabilization & Recovery (t + 72h)
- Sector-based risk assessment derived from actual ML susceptibility & observed flood data
- Resource budget allocation (Rescue Boats, Sandbag Barriers, Mobile Medical Units, Food/Water Kits)
- Quantitative scoring engine (Lives Protected, Resource Efficiency, Budget Adherence, False Alarms)
- Detailed post-mission AAR (After-Action Report) with tactical recommendations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class SectorInfo:
    id: str
    name: str
    lat: float
    lon: float
    population: int
    critical_infra: str
    base_susceptibility: float
    observed_flooded: bool
    terrain_desc: str


# Documented critical sectors across the Kaziranga-Jorhat-Brahmaputra corridor
KEY_SECTORS: Dict[str, SectorInfo] = {
    "SECTOR_NORTH_KAZIRANGA": SectorInfo(
        id="SECTOR_NORTH_KAZIRANGA",
        name="Northern Kaziranga Lowlands",
        lat=26.68,
        lon=93.25,
        population=42000,
        critical_infra="Wildlife Sanctuary Edge & Rural Settlements",
        base_susceptibility=0.88,
        observed_flooded=True,
        terrain_desc="Low floodplain (<65m), slope <0.3°",
    ),
    "SECTOR_BOKAKHAT": SectorInfo(
        id="SECTOR_BOKAKHAT",
        name="Bokakhat Sub-Division Corridor",
        lat=26.63,
        lon=93.59,
        population=68000,
        critical_infra="National Highway 715 & Regional Hospital",
        base_susceptibility=0.74,
        observed_flooded=True,
        terrain_desc="Terrace (65-75m), key transport lifeline",
    ),
    "SECTOR_GOLAGHAT": SectorInfo(
        id="SECTOR_GOLAGHAT",
        name="Golaghat Urban & Agriculture Fringe",
        lat=26.52,
        lon=93.97,
        population=85000,
        critical_infra="Power Substation & Grain Storage Depots",
        base_susceptibility=0.48,
        observed_flooded=False,
        terrain_desc="Terrace (75-85m), moderate drainage",
    ),
    "SECTOR_JORHAT_WEST": SectorInfo(
        id="SECTOR_JORHAT_WEST",
        name="Western Jorhat Embankment",
        lat=26.74,
        lon=94.15,
        population=95000,
        critical_infra="Brahmaputra River Embankment & Ferry Ghat",
        base_susceptibility=0.82,
        observed_flooded=True,
        terrain_desc="Direct riverfront, high backwater surge",
    ),
    "SECTOR_TEZPUR_EAST": SectorInfo(
        id="SECTOR_TEZPUR_EAST",
        name="Eastern Tezpur Foothills",
        lat=26.65,
        lon=92.85,
        population=52000,
        critical_infra="Clean Water Treatment Plant & Relief Center",
        base_susceptibility=0.31,
        observed_flooded=False,
        terrain_desc="Piedmont (85-110m), naturally elevated",
    ),
}


RESOURCE_COSTS = {
    "rescue_teams": 12000,       # USD / INR relative points per team
    "flood_barriers": 18000,     # per km deployed
    "medical_units": 15000,      # per deployed clinic
    "relief_rations": 8000,      # per 10k population supply
}


@dataclass
class MissionState:
    event_id: str = "DFO_4259 (Brahmaputra Surge)"
    current_phase: int = 1       # 1 to 4
    total_budget: int = 150000   # Starting resource budget
    spent_budget: int = 0
    resources_allocated: Dict[str, Dict[str, int]] = field(default_factory=dict)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    mission_complete: bool = False
    final_score: int = 0
    eval_report: Dict[str, Any] = field(default_factory=dict)


class FloodCommanderSim:
    """
    Simulation engine managing game state, phases, and evaluation.
    """

    def __init__(self, state: Optional[MissionState] = None) -> None:
        self.state = state or MissionState()
        if not self.state.resources_allocated:
            # Initialize empty sector allocations
            for s_id in KEY_SECTORS:
                self.state.resources_allocated[s_id] = {
                    "rescue_teams": 0,
                    "flood_barriers": 0,
                    "medical_units": 0,
                    "relief_rations": 0,
                }

    def allocate_resource(self, sector_id: str, resource_type: str, count: int) -> Tuple[bool, str]:
        """Allocate or remove resources for a given sector."""
        if self.state.mission_complete:
            return False, "Mission is already completed."

        current_count = self.state.resources_allocated[sector_id].get(resource_type, 0)
        cost_per_unit = RESOURCE_COSTS.get(resource_type, 10000)
        cost_diff = (count - current_count) * cost_per_unit

        if self.state.spent_budget + cost_diff > self.state.total_budget:
            return False, f"Insufficient budget! Additional cost is ${cost_diff:,}, but only ${(self.state.total_budget - self.state.spent_budget):,} remains."

        self.state.resources_allocated[sector_id][resource_type] = count
        self.state.spent_budget += cost_diff
        return True, "Allocation updated."

    def advance_phase(self) -> Dict[str, Any]:
        """Advance to the next tactical phase of the crisis."""
        if self.state.current_phase < 4:
            self.state.current_phase += 1
            phase_narratives = {
                2: "Phase 2: Torrential rainfall triggers upstream runoff. Water gauge at Tezpur exceeds danger mark.",
                3: "Phase 3: Brahmaputra crests at peak discharge. Embankment stress reported across northern and eastern sectors.",
                4: "Phase 4: Waters begin recession. Secondary epidemic risk and stranded citizen relief operations underway.",
            }
            return {"advanced": True, "phase": self.state.current_phase, "message": phase_narratives.get(self.state.current_phase, "")}
        else:
            self.state.mission_complete = True
            report = self.compute_evaluation()
            self.state.eval_report = report
            self.state.final_score = report["overall_score"]
            return {"advanced": False, "phase": 4, "message": "Mission complete. Generating final evaluation report."}

    def compute_evaluation(self) -> Dict[str, Any]:
        """
        Evaluate commander decisions against actual flood ground truth and ML susceptibility.
        """
        score = 0
        total_possible = 100
        sector_evals = []
        strengths = []
        mistakes = []
        recommendations = []

        total_vulnerable_pop_protected = 0
        wasted_barriers_on_dry_sectors = 0

        for s_id, sector in KEY_SECTORS.items():
            allocs = self.state.resources_allocated.get(s_id, {})
            r_teams = allocs.get("rescue_teams", 0)
            barriers = allocs.get("flood_barriers", 0)
            meds = allocs.get("medical_units", 0)
            rations = allocs.get("relief_rations", 0)

            is_flooded = sector.observed_flooded
            susc = sector.base_susceptibility

            # Evaluation metrics for this sector
            sector_score = 0
            feedback = []

            if is_flooded:
                # Needed defenses and rescue
                if barriers >= 1:
                    sector_score += 8
                    feedback.append("Defensive barriers successfully mitigated initial wave impact.")
                else:
                    mistakes.append(f"No barriers placed in high-risk flooded zone: {sector.name}.")

                if r_teams >= 1:
                    protected = int(sector.population * min(1.0, 0.4 * r_teams))
                    total_vulnerable_pop_protected += protected
                    sector_score += 10
                    feedback.append(f"Deployed rescue teams successfully evacuated estimated {protected:,} residents.")
                else:
                    mistakes.append(f"Critical lack of rescue teams in heavily inundated {sector.name} ({sector.population:,} pop).")

                if meds >= 1 and rations >= 1:
                    sector_score += 7
                    feedback.append("Post-flood rations and medical response fully operational.")
                else:
                    feedback.append("Relief supplies under-supplied for inundated population.")
            else:
                # Dry / unflooded sector
                if barriers > 1:
                    wasted_barriers_on_dry_sectors += barriers
                    sector_score -= 5
                    mistakes.append(f"Over-allocation: Wasted {barriers} defensive barriers in non-flooded {sector.name}.")
                else:
                    sector_score += 10
                    feedback.append("Optimal resource conservation: correctly avoided over-fortifying dry sector.")

                if r_teams == 0 and susc < 0.40:
                    strengths.append(f"Prudent resource discipline in low-susceptibility {sector.name}.")

            sector_evals.append({
                "sector_name": sector.name,
                "was_flooded": is_flooded,
                "model_susceptibility": susc,
                "resources": allocs,
                "sector_score": sector_score,
                "feedback": " ".join(feedback),
            })
            score += sector_score

        # Budget efficiency score
        rem_budget = self.state.total_budget - self.state.spent_budget
        budget_ratio = self.state.spent_budget / max(1, self.state.total_budget)
        if 0.70 <= budget_ratio <= 0.98:
            score += 15
            strengths.append(f"Prudent fiscal management: Deployed 70-98% of available capital (${self.state.spent_budget:,} spent).")
        elif budget_ratio < 0.50:
            mistakes.append("Severe under-utilization of response budget: Emergency capital remained idle during crisis.")
        else:
            score += 5

        # Normalize score to 0 - 100
        overall_score = max(0, min(100, int(score)))

        if overall_score >= 80:
            rank = "Exemplary Commander (Class A)"
            recommendations.append("Your deployment demonstrated strong alignment with geospatial risk priors and rapid tactical response.")
        elif overall_score >= 60:
            rank = "Proficient Officer (Class B)"
            recommendations.append("Good coverage of major floodways, but improve reconnaissance to avoid deploying structural defenses to dry sectors.")
        else:
            rank = "Needs Strategic Review (Class C)"
            recommendations.append("Rely more directly on the Neural Network and Random Forest susceptibility layers to pre-stage supplies prior to inundation.")

        return {
            "overall_score": overall_score,
            "rank": rank,
            "budget_spent": self.state.spent_budget,
            "budget_remaining": rem_budget,
            "citizens_protected": total_vulnerable_pop_protected,
            "sector_details": sector_evals,
            "strengths": strengths,
            "mistakes": mistakes,
            "recommendations": recommendations,
        }

    def reset(self) -> None:
        """Reset mission to beginning."""
        self.state = MissionState()
        for s_id in KEY_SECTORS:
            self.state.resources_allocated[s_id] = {
                "rescue_teams": 0,
                "flood_barriers": 0,
                "medical_units": 0,
                "relief_rations": 0,
            }
