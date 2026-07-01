"""Transparent cooling intervention scenario simulation utilities.

The default response model is intentionally conservative. It estimates
directional cooling from existing model outputs and explanatory signals rather
than claiming a calibrated intervention physics model. The `InterventionResponseModel`
protocol is the replacement point for a future PINN or other physically
constrained simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import numpy as np
import pandas as pd


INTERVENTION_URBAN_GREENING = "Urban Greening"
INTERVENTION_COOL_ROOFS = "Cool Roofs"
INTERVENTION_HIGH_ALBEDO = "Reflective / High-Albedo Surfaces"
INTERVENTION_BLUE_GREEN = "Blue-Green Infrastructure"
INTERVENTION_WATER_RESTORATION = "Water Body Restoration"

SUPPORTED_INTERVENTIONS = (
    INTERVENTION_URBAN_GREENING,
    INTERVENTION_COOL_ROOFS,
    INTERVENTION_HIGH_ALBEDO,
    INTERVENTION_BLUE_GREEN,
    INTERVENTION_WATER_RESTORATION,
)


@dataclass(frozen=True)
class InterventionAssumption:
    """Domain assumptions for one intervention type.

    `max_cooling_c` is a practical upper bound for a broad ward-scale scenario at
    full intensity and coverage. It is deliberately below extreme site-scale
    microclimate claims because dashboard outputs are ward-level planning
    estimates, not street-canyon CFD results.
    """

    label: str
    max_cooling_c: float
    primary_features: tuple[str, ...]
    sensitivity_feature: str
    notes: tuple[str, ...]


INTERVENTION_ASSUMPTIONS: dict[str, InterventionAssumption] = {
    INTERVENTION_URBAN_GREENING: InterventionAssumption(
        label=INTERVENTION_URBAN_GREENING,
        max_cooling_c=2.4,
        primary_features=("NDVI", "DW_TREES_PROB", "DW_GRASS_PROB"),
        sensitivity_feature="NDVI",
        notes=(
            "Cooling increases where vegetation cover is currently low and heat stress is high.",
            "Response is bounded to ward-scale LST change, not local shade temperature.",
        ),
    ),
    INTERVENTION_COOL_ROOFS: InterventionAssumption(
        label=INTERVENTION_COOL_ROOFS,
        max_cooling_c=1.6,
        primary_features=("BUILT_FRACTION", "BUILT_NRES_FRACTION", "NDBI"),
        sensitivity_feature="BUILT_FRACTION",
        notes=(
            "Most suitable where built fraction and built-up spectral intensity are high.",
            "Assumes roof-surface treatment only; it does not cool open water or vegetated land.",
        ),
    ),
    INTERVENTION_HIGH_ALBEDO: InterventionAssumption(
        label=INTERVENTION_HIGH_ALBEDO,
        max_cooling_c=1.3,
        primary_features=("ALBEDO", "NDBI", "BUILT_FRACTION"),
        sensitivity_feature="ALBEDO",
        notes=(
            "Benefit is larger where current albedo is low and impervious area is high.",
            "The model caps impact to avoid impossible temperature reductions from reflectance alone.",
        ),
    ),
    INTERVENTION_BLUE_GREEN: InterventionAssumption(
        label=INTERVENTION_BLUE_GREEN,
        max_cooling_c=2.0,
        primary_features=("NDVI", "DW_WATER_PROB", "S2_MNDWI"),
        sensitivity_feature="NDVI",
        notes=(
            "Combines vegetation, infiltration, and small water features for heat-stress reduction.",
            "Suitable for low-vegetation wards where drainage or open-space retrofits are plausible.",
        ),
    ),
    INTERVENTION_WATER_RESTORATION: InterventionAssumption(
        label=INTERVENTION_WATER_RESTORATION,
        max_cooling_c=1.8,
        primary_features=("DW_WATER_PROB", "S2_MNDWI"),
        sensitivity_feature="DW_WATER_PROB",
        notes=(
            "Applies only as restoration or enhancement of plausible blue-space opportunity.",
            "Confidence drops when the current water signal is absent because siting feasibility is unknown.",
        ),
    ),
}


@dataclass(frozen=True)
class ScenarioInput:
    ward: str
    intervention_type: str
    intensity: float
    coverage_pct: float


@dataclass(frozen=True)
class ScenarioResult:
    ward_id: str
    ward_name: str
    intervention_type: str
    intensity: float
    coverage_pct: float
    estimated_cooling_c: float
    affected_area_km2: float
    confidence: str
    confidence_score: float
    implementation_notes: tuple[str, ...]
    drivers_used: tuple[str, ...]
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ward_id": self.ward_id,
            "ward_name": self.ward_name,
            "intervention_type": self.intervention_type,
            "intensity": self.intensity,
            "coverage_pct": self.coverage_pct,
            "estimated_cooling_c": self.estimated_cooling_c,
            "affected_area_km2": self.affected_area_km2,
            "confidence": self.confidence,
            "confidence_score": self.confidence_score,
            "implementation_notes": list(self.implementation_notes),
            "drivers_used": list(self.drivers_used),
            "assumptions": list(self.assumptions),
        }


class InterventionResponseModel(Protocol):
    def estimate(
        self,
        ward_record: Mapping[str, object],
        intervention_type: str,
        intensity: float,
        coverage_pct: float,
        shap_summary: pd.DataFrame | None = None,
    ) -> ScenarioResult:
        ...


@dataclass
class DefaultInterventionResponseModel:
    assumptions: Mapping[str, InterventionAssumption] = field(default_factory=lambda: INTERVENTION_ASSUMPTIONS)

    def estimate(
        self,
        ward_record: Mapping[str, object],
        intervention_type: str,
        intensity: float,
        coverage_pct: float,
        shap_summary: pd.DataFrame | None = None,
    ) -> ScenarioResult:
        if intervention_type not in self.assumptions:
            raise ValueError(f"Unsupported intervention type: {intervention_type}")

        bounded_intensity = _bound(float(intensity), 0.0, 100.0) / 100.0
        bounded_coverage = _bound(float(coverage_pct), 0.0, 100.0) / 100.0
        assumption = self.assumptions[intervention_type]

        heat_multiplier = _heat_stress_multiplier(ward_record)
        suitability = _intervention_suitability(ward_record, intervention_type)
        shap_support = _shap_support(assumption.primary_features, shap_summary)
        raw_cooling = assumption.max_cooling_c * bounded_intensity * bounded_coverage * suitability * heat_multiplier
        estimated_cooling = round(_bound(raw_cooling, 0.0, assumption.max_cooling_c), 2)

        affected_area = round(_estimate_area_km2(ward_record) * bounded_coverage, 3)
        confidence_score = _confidence_score(ward_record, coverage_pct, shap_support, suitability, intervention_type)
        confidence = _confidence_label(confidence_score)

        ward_id = str(_first_present(ward_record, ("ward_id", "WARD", "ward", "id"), "Citywide"))
        ward_name = str(_first_present(ward_record, ("ward_name", "WARD", "ward", "name"), f"Ward {ward_id}"))
        drivers = tuple(feature for feature in assumption.primary_features if _has_number(ward_record, feature))
        notes = _implementation_notes(ward_record, assumption, suitability, shap_support, estimated_cooling)

        return ScenarioResult(
            ward_id=ward_id,
            ward_name=ward_name,
            intervention_type=intervention_type,
            intensity=round(bounded_intensity * 100.0, 1),
            coverage_pct=round(bounded_coverage * 100.0, 1),
            estimated_cooling_c=estimated_cooling,
            affected_area_km2=affected_area,
            confidence=confidence,
            confidence_score=round(confidence_score, 2),
            implementation_notes=notes,
            drivers_used=drivers,
            assumptions=assumption.notes,
        )


def simulate_intervention(
    ward_record: Mapping[str, object],
    intervention_type: str,
    intensity: float,
    coverage_pct: float,
    shap_summary: pd.DataFrame | None = None,
    response_model: InterventionResponseModel | None = None,
) -> ScenarioResult:
    model = response_model or DefaultInterventionResponseModel()
    return model.estimate(ward_record, intervention_type, intensity, coverage_pct, shap_summary)


def simulate_for_wards(
    ward_records: pd.DataFrame,
    intervention_type: str,
    intensity: float,
    coverage_pct: float,
    shap_summary: pd.DataFrame | None = None,
    response_model: InterventionResponseModel | None = None,
) -> pd.DataFrame:
    if ward_records.empty:
        return pd.DataFrame()
    results = [
        simulate_intervention(row._asdict(), intervention_type, intensity, coverage_pct, shap_summary, response_model).to_dict()
        for row in ward_records.itertuples(index=False)
    ]
    return pd.DataFrame(results)


def _intervention_suitability(ward_record: Mapping[str, object], intervention_type: str) -> float:
    ndvi = _normalized(_number(ward_record, "NDVI"), -0.1, 0.6, default=0.45)
    ndbi = _normalized(_number(ward_record, "NDBI"), -0.3, 0.35, default=0.5)
    built = _normalized(_number(ward_record, "BUILT_FRACTION"), 0.0, 0.85, default=0.5)
    albedo = _normalized(_number(ward_record, "ALBEDO"), 0.08, 0.28, default=0.45)
    water = _normalized(_number(ward_record, "DW_WATER_PROB"), 0.0, 0.35, default=0.25)

    if intervention_type == INTERVENTION_URBAN_GREENING:
        return _bound(0.25 + 0.55 * (1.0 - ndvi) + 0.20 * built, 0.25, 1.0)
    if intervention_type == INTERVENTION_COOL_ROOFS:
        return _bound(0.20 + 0.55 * built + 0.25 * ndbi, 0.2, 1.0)
    if intervention_type == INTERVENTION_HIGH_ALBEDO:
        return _bound(0.20 + 0.45 * (1.0 - albedo) + 0.35 * built, 0.2, 1.0)
    if intervention_type == INTERVENTION_BLUE_GREEN:
        return _bound(0.25 + 0.35 * (1.0 - ndvi) + 0.25 * built + 0.15 * (1.0 - water), 0.25, 1.0)
    if intervention_type == INTERVENTION_WATER_RESTORATION:
        restoration_opportunity = 1.0 - abs(water - 0.30) / 0.70
        return _bound(0.18 + 0.45 * restoration_opportunity + 0.20 * (1.0 - ndvi) + 0.17 * built, 0.18, 0.85)
    return 0.3


def _heat_stress_multiplier(ward_record: Mapping[str, object]) -> float:
    lst = _first_number(ward_record, ("mean_lst", "PREDICTED_LST_C", "LST_C"))
    if lst is None:
        return 1.0
    return _bound(0.75 + max(0.0, lst - 34.0) / 8.0, 0.75, 1.25)


def _shap_support(features: tuple[str, ...], shap_summary: pd.DataFrame | None) -> float:
    if shap_summary is None or shap_summary.empty or "feature" not in shap_summary:
        return 0.0
    top_features = shap_summary.head(8)["feature"].astype(str).tolist()
    if not top_features:
        return 0.0
    matches = sum(1 for feature in features if feature in top_features)
    return matches / max(1, min(len(features), 3))


def _confidence_score(
    ward_record: Mapping[str, object],
    coverage_pct: float,
    shap_support: float,
    suitability: float,
    intervention_type: str,
) -> float:
    evidence = 0.38
    if _first_number(ward_record, ("sample_count", "pixel_count")):
        evidence += 0.18
    if _first_number(ward_record, ("mean_lst", "PREDICTED_LST_C", "LST_C")):
        evidence += 0.12
    evidence += 0.18 * _bound(shap_support, 0.0, 1.0)
    evidence += 0.10 * suitability
    if coverage_pct > 60:
        evidence -= 0.10
    if intervention_type == INTERVENTION_WATER_RESTORATION and _number(ward_record, "DW_WATER_PROB") is None:
        evidence -= 0.10
    return _bound(evidence, 0.15, 0.9)


def _confidence_label(score: float) -> str:
    if score >= 0.72:
        return "High"
    if score >= 0.48:
        return "Medium"
    return "Low"


def _implementation_notes(
    ward_record: Mapping[str, object],
    assumption: InterventionAssumption,
    suitability: float,
    shap_support: float,
    estimated_cooling: float,
) -> tuple[str, ...]:
    notes = [
        f"Estimated cooling is bounded by a {assumption.max_cooling_c:.1f} C ward-scale maximum for {assumption.label}.",
        f"Suitability score from current ward features is {suitability:.2f}.",
    ]
    if shap_support > 0:
        notes.append("SHAP importance supports at least one intervention-relevant driver.")
    else:
        notes.append("SHAP support was unavailable or indirect; treat this as a directional estimate.")
    if estimated_cooling <= 0.05:
        notes.append("Projected effect is near zero at the selected intensity and coverage.")
    lst = _first_number(ward_record, ("mean_lst", "PREDICTED_LST_C", "LST_C"))
    if lst is not None:
        notes.append(f"Existing model heat signal used: {lst:.2f} C ward-level LST.")
    return tuple(notes)


def _estimate_area_km2(ward_record: Mapping[str, object]) -> float:
    area = _first_number(ward_record, ("area_km2", "ward_area_km2"))
    if area is not None and area > 0:
        return area
    pixel_count = _first_number(ward_record, ("pixel_count", "sample_count"))
    if pixel_count is not None and pixel_count > 0:
        return float(pixel_count) * 900.0 / 1_000_000.0
    return 1.0


def _first_present(ward_record: Mapping[str, object], keys: tuple[str, ...], default: object) -> object:
    for key in keys:
        value = ward_record.get(key)
        if value is not None and not pd.isna(value):
            return value
    return default


def _first_number(ward_record: Mapping[str, object], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _number(ward_record, key)
        if value is not None:
            return value
    return None


def _has_number(ward_record: Mapping[str, object], key: str) -> bool:
    return _number(ward_record, key) is not None


def _number(ward_record: Mapping[str, object], key: str) -> float | None:
    value = ward_record.get(key)
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized(value: float | None, low: float, high: float, default: float) -> float:
    if value is None or high <= low:
        return default
    return _bound((value - low) / (high - low), 0.0, 1.0)


def _bound(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))
