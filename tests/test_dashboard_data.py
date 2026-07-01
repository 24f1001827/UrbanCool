from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from src.dashboard.data import (
    DashboardArtifacts,
    derive_hotspot_metrics,
    derive_ward_summary,
    get_demo_status,
    load_ward_boundaries,
    summarize_global_shap,
)


class DashboardDataTests(unittest.TestCase):
    def test_derive_hotspot_metrics_is_deterministic(self) -> None:
        points = pd.DataFrame(
            {
                "latitude": [22.50, 22.51, 22.52, 22.53],
                "longitude": [88.30, 88.31, 88.32, 88.33],
                "LST_C": [31.0, 32.0, 35.0, 39.0],
            }
        )

        derived = derive_hotspot_metrics(points)

        self.assertIn("hotspot_label", derived.columns)
        self.assertTrue(derived["hotspot_score"].between(0, 100).all())
        self.assertEqual(derived.sort_values("LST_C")["hotspot_label"].tolist()[-1], "Severe")

    def test_get_demo_status_handles_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = DashboardArtifacts(
                training_sample_path=root / "missing_points.csv",
                shap_values_path=root / "missing_shap.csv",
                ward_boundaries_path=root / "missing_wards.geojson",
                ward_summary_path=root / "missing_summary.csv",
            )

            status = get_demo_status(artifacts)

            self.assertFalse(status["training_points_ready"])
            self.assertFalse(status["shap_ready"])
            self.assertFalse(status["ward_boundaries_ready"])
            self.assertFalse(status["ward_summary_ready"])

    def test_generated_ward_summary_counts_as_ward_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outputs_dir = root / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            gdf = gpd.GeoDataFrame(
                {
                    "WARD": ["93"],
                    "PREDICTED_LST_C": [36.8],
                    "NDVI": [0.32],
                    "NDBI": [-0.03],
                    "BUILT_FRACTION": [0.46],
                    "DW_WATER_PROB": [0.07],
                    "pixel_count": [2242],
                    "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                },
                crs="EPSG:4326",
            )
            gdf.to_file(outputs_dir / "ward_heat_summary.geojson", driver="GeoJSON")

            cwd = Path.cwd()
            os.chdir(root)
            try:
                wards = load_ward_boundaries()
                status = get_demo_status()
            finally:
                os.chdir(cwd)

            self.assertFalse(wards.empty)
            self.assertEqual(wards.iloc[0]["ward_name"], "Ward 93")
            self.assertTrue(status["ward_boundaries_ready"])
            self.assertTrue(status["ward_summary_ready"])

    def test_derive_ward_summary_adds_rank_and_driver(self) -> None:
        points = pd.DataFrame(
            {
                "latitude": [0.1, 0.2, 1.1, 1.2],
                "longitude": [0.1, 0.2, 1.1, 1.2],
                "LST_C": [30.0, 31.0, 37.0, 38.0],
            }
        )
        shap_values = pd.DataFrame(
            {
                "NDVI": [-0.3, -0.2, -0.1, -0.1],
                "NDBI": [0.1, 0.2, 0.8, 0.9],
                "prediction": [30.5, 31.2, 36.8, 38.2],
                "LST_C": [30.0, 31.0, 37.0, 38.0],
            }
        )
        wards = gpd.GeoDataFrame(
            {
                "ward_no": ["1", "2"],
                "name": ["North", "South"],
                "geometry": [
                    Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]),
                    Polygon([(1, 1), (1.5, 1), (1.5, 1.5), (1, 1.5)]),
                ],
            },
            crs="EPSG:4326",
        )

        summary = derive_ward_summary(points, wards, shap_values)

        self.assertEqual(len(summary), 2)
        self.assertIn("rank", summary.columns)
        self.assertIn("dominant_driver", summary.columns)
        south = summary[summary["ward_name"] == "South"].iloc[0]
        self.assertEqual(south["dominant_driver"], "NDBI")
        self.assertEqual(int(south["rank"]), 1)

    def test_summarize_global_shap_orders_features(self) -> None:
        shap_values = pd.DataFrame(
            {
                "NDVI": [0.1, 0.2, 0.1],
                "NDBI": [0.9, 0.8, 0.7],
                "prediction": [1, 2, 3],
                "LST_C": [1, 2, 3],
            }
        )

        summary = summarize_global_shap(shap_values)

        self.assertEqual(summary.iloc[0]["feature"], "NDBI")


if __name__ == "__main__":
    unittest.main()
