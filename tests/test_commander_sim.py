"""
tests/test_commander_sim.py

Unit tests for Flood Commander simulation engine:
- Sector initialization
- Resource allocation limits and budget tracking
- Timeline advancement across disaster phases
- Ground-truth evaluation against real historical flood observations
- Scoring logic and post-mission report generation
"""

import pytest
from src.simulation.disaster_commander import FloodCommanderSim, KEY_SECTORS, RESOURCE_COSTS


def test_commander_initialization():
    sim = FloodCommanderSim()
    assert sim.state.current_phase == 1
    assert sim.state.total_budget == 150000
    assert sim.state.spent_budget == 0
    assert not sim.state.mission_complete
    assert len(sim.state.resources_allocated) == len(KEY_SECTORS)


def test_resource_allocation_and_budget_bounds():
    sim = FloodCommanderSim()
    sector_id = "SECTOR_NORTH_KAZIRANGA"

    # Allocate 2 rescue teams ($24,000)
    ok, msg = sim.allocate_resource(sector_id, "rescue_teams", 2)
    assert ok
    assert sim.state.spent_budget == 24000
    assert sim.state.resources_allocated[sector_id]["rescue_teams"] == 2

    # Attempt to exceed remaining budget with 20 flood barriers ($360,000)
    ok_exceed, msg_exceed = sim.allocate_resource(sector_id, "flood_barriers", 20)
    assert not ok_exceed
    assert "Insufficient budget" in msg_exceed
    # Budget spent should remain unchanged
    assert sim.state.spent_budget == 24000


def test_simulation_phase_advancement_and_evaluation():
    sim = FloodCommanderSim()

    # Pre-stage resources on high risk flooded sectors
    sim.allocate_resource("SECTOR_NORTH_KAZIRANGA", "rescue_teams", 2)
    sim.allocate_resource("SECTOR_NORTH_KAZIRANGA", "flood_barriers", 2)
    sim.allocate_resource("SECTOR_JORHAT_WEST", "rescue_teams", 2)
    sim.allocate_resource("SECTOR_JORHAT_WEST", "flood_barriers", 1)

    # Advance through all 4 phases
    res1 = sim.advance_phase()
    assert res1["phase"] == 2
    assert not sim.state.mission_complete

    res2 = sim.advance_phase()
    assert res2["phase"] == 3

    res3 = sim.advance_phase()
    assert res3["phase"] == 4

    # Final advance completes mission
    res_final = sim.advance_phase()
    assert sim.state.mission_complete
    assert sim.state.final_score > 0
    assert "overall_score" in sim.state.eval_report
    assert "citizens_protected" in sim.state.eval_report
    assert len(sim.state.eval_report["strengths"]) > 0 or len(sim.state.eval_report["mistakes"]) > 0


def test_reset_functionality():
    sim = FloodCommanderSim()
    sim.allocate_resource("SECTOR_BOKAKHAT", "rescue_teams", 3)
    sim.advance_phase()
    assert sim.state.current_phase > 1
    assert sim.state.spent_budget > 0

    sim.reset()
    assert sim.state.current_phase == 1
    assert sim.state.spent_budget == 0
    assert not sim.state.mission_complete
