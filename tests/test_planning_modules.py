from __future__ import annotations

import unittest

import pandas as pd

from src.llm_layer import generate_city_briefing, generate_ward_summary
from src.optimization import optimize_cooling_plan
from src.scenario_engine import SUPPORTED_INTERVENTIONS, simulate_intervention


class PlanningModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wards = pd.DataFrame(
            {
                "ward_id": ["37", "12"],
                "ward_name": ["Ward 37", "Ward 12"],
                "PREDICTED_LST_C": [39.2, 36.5],
                "mean_lst": [39.2, 36.5],
                "NDVI": [0.18, 0.34],
                "NDBI": [0.08, -0.02],
                "BUILT_FRACTION": [0.58, 0.32],
                "DW_WATER_PROB": [0.04, 0.12],
                "pixel_count": [1000, 900],
                "hotspot_score": [92.0, 55.0],
            }
        )
        self.shap_summary = pd.DataFrame(
            {
                "feature": ["BUILT_NRES_FRACTION", "NDBI", "DW_WATER_PROB", "NDVI"],
                "mean_abs_shap": [0.7, 0.5, 0.4, 0.3],
            }
        )

    def test_scenario_output_is_bounded_and_transparent(self) -> None:
        result = simulate_intervention(self.wards.iloc[0].to_dict(), "Urban Greening", 80, 30, self.shap_summary)

        self.assertGreater(result.estimated_cooling_c, 0)
        self.assertLessEqual(result.estimated_cooling_c, 2.4)
        self.assertGreater(result.affected_area_km2, 0)
        self.assertIn(result.confidence, {"Low", "Medium", "High"})
        self.assertTrue(result.implementation_notes)

    def test_optimizer_respects_budget(self) -> None:
        result = optimize_cooling_plan(
            self.wards,
            total_budget=120_000_000,
            target_wards=["Ward 37", "Ward 12"],
            shap_summary=self.shap_summary,
        )

        self.assertLessEqual(result.allocated_budget_inr, result.total_budget_inr)
        self.assertIn(result.strategy, {"greedy_baseline"})
        self.assertFalse(result.candidate_table.empty)

    def test_llm_templates_are_deterministic(self) -> None:
        brief = generate_ward_summary(self.wards.iloc[0], self.shap_summary)
        city = generate_city_briefing(self.wards, optimization_plan=pd.DataFrame())

        self.assertIn("Ward 37", brief.summary)
        self.assertIn("expected", brief.recommendation)
        self.assertIn("Across 2 mapped wards", city)

    def test_all_supported_interventions_can_simulate(self) -> None:
        outputs = [
            simulate_intervention(self.wards.iloc[0].to_dict(), intervention, 60, 20, self.shap_summary)
            for intervention in SUPPORTED_INTERVENTIONS
        ]

        self.assertEqual(len(outputs), 5)
        self.assertTrue(all(output.estimated_cooling_c >= 0 for output in outputs))


if __name__ == "__main__":
    unittest.main()
