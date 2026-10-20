"""
src/models/fuzzy_layer.py

Mamdani Fuzzy Inference System for Post-Processing & Interpretable Risk Categorization.
Inputs:
1. Model Susceptibility Probability [0.0 to 1.0]
2. Antecedent 3-Day Rainfall (mm) [0.0 to 150.0 mm]

Output:
- Fuzzy Risk Score [0.0 to 100.0] mapped to:
  - Low Risk (< 35)
  - Moderate Risk (35 - 70)
  - High Risk (> 70)

NOTE: As per project guidelines, this fuzzy layer is designed strictly for
transparent human interpretability and operational rule-based decision support,
NOT for claiming superior numerical accuracy over the underlying neural network.
"""

from typing import Tuple, Dict
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class MamdaniRiskFuzzySystem:
    def __init__(self):
        # 1. Antecedents (Inputs)
        self.prob = ctrl.Antecedent(np.linspace(0.0, 1.0, 101), "probability")
        self.rain = ctrl.Antecedent(np.linspace(0.0, 150.0, 151), "rainfall_3d")

        # 2. Consequent (Output)
        self.risk = ctrl.Consequent(np.linspace(0.0, 100.0, 101), "risk_score")

        # 3. Membership Functions
        # Probability memberships
        self.prob["low"] = fuzz.trapmf(self.prob.universe, [0.0, 0.0, 0.25, 0.45])
        self.prob["medium"] = fuzz.trimf(self.prob.universe, [0.30, 0.55, 0.75])
        self.prob["high"] = fuzz.trapmf(self.prob.universe, [0.60, 0.80, 1.0, 1.0])

        # Rainfall 3-day memberships (mm)
        self.rain["low"] = fuzz.trapmf(self.rain.universe, [0.0, 0.0, 20.0, 45.0])
        self.rain["moderate"] = fuzz.trimf(self.rain.universe, [30.0, 60.0, 95.0])
        self.rain["heavy"] = fuzz.trapmf(self.rain.universe, [75.0, 110.0, 150.0, 150.0])

        # Risk score output memberships
        self.risk["low"] = fuzz.trapmf(self.risk.universe, [0.0, 0.0, 20.0, 40.0])
        self.risk["moderate"] = fuzz.trimf(self.risk.universe, [30.0, 50.0, 75.0])
        self.risk["high"] = fuzz.trapmf(self.risk.universe, [65.0, 85.0, 100.0, 100.0])

        # 4. Fuzzy Rule Base (~8 rules)
        rules = [
            # Rule 1: High probability + heavy rain -> High risk
            ctrl.Rule(self.prob["high"] & self.rain["heavy"], self.risk["high"]),
            # Rule 2: High probability + moderate rain -> High risk
            ctrl.Rule(self.prob["high"] & self.rain["moderate"], self.risk["high"]),
            # Rule 3: High probability + low rain -> Moderate risk
            ctrl.Rule(self.prob["high"] & self.rain["low"], self.risk["moderate"]),
            # Rule 4: Medium probability + heavy rain -> High risk
            ctrl.Rule(self.prob["medium"] & self.rain["heavy"], self.risk["high"]),
            # Rule 5: Medium probability + moderate rain -> Moderate risk
            ctrl.Rule(self.prob["medium"] & self.rain["moderate"], self.risk["moderate"]),
            # Rule 6: Medium probability + low rain -> Low risk
            ctrl.Rule(self.prob["medium"] & self.rain["low"], self.risk["low"]),
            # Rule 7: Low probability + heavy rain -> Moderate risk (flash ponding potential)
            ctrl.Rule(self.prob["low"] & self.rain["heavy"], self.risk["moderate"]),
            # Rule 8: Low probability + low/moderate rain -> Low risk
            ctrl.Rule(self.prob["low"] & ~self.rain["heavy"], self.risk["low"]),
        ]

        # 5. Control Simulation System
        risk_ctrl = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(risk_ctrl)

    def evaluate(self, probability: float, rainfall_3d: float) -> Tuple[float, str]:
        """
        Evaluate a single point through the Mamdani fuzzy system.
        Returns (defuzzified_risk_score [0-100], linguistic_category).
        """
        prob_clamped = float(np.clip(probability, 0.0, 1.0))
        rain_clamped = float(np.clip(rainfall_3d, 0.0, 150.0))

        self.sim.input["probability"] = prob_clamped
        self.sim.input["rainfall_3d"] = rain_clamped

        try:
            self.sim.compute()
            score = float(self.sim.output["risk_score"])
        except Exception:
            # Fallback heuristic if out-of-bounds
            score = prob_clamped * 100.0

        if score >= 70.0:
            category = "High Risk"
        elif score >= 35.0:
            category = "Moderate Risk"
        else:
            category = "Low Risk"

        return round(score, 1), category
