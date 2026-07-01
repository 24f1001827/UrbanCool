"""Deterministic planner-facing explanation helpers.

This module is intentionally named as an LLM layer because it owns the natural
language decision support surface, but it has no external API dependency. The
templates are deterministic so demos remain reproducible offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FEATURE_LABELS = {
    "LST_C": "Land surface temperature",
    "NDVI": "Vegetation cover",
    "NDBI": "Built-up intensity",
    "BUILT_FRACTION": "Built fraction",
    "BUILT_NRES_FRACTION": "Non-residential built fraction",
    "MEAN_HEIGHT_150M": "Mean building height",
    "CANYON_ASPECT_RATIO_HW": "Canyon aspect ratio",
    "SKY_VIEW_FACTOR": "Sky view factor",
    "DW_WATER_PROB": "Water presence",
    "NET_SOLAR_RADIATION_W_M2": "Net solar radiation",
    "ALBEDO": "Surface albedo",
}


@dataclass(frozen=True)
class WardBrief:
    ward_id: str
    ward_name: str
    summary: str
    recommendation: str
    evidence: tuple[str, ...]


def generate_ward_summary(
    ward_record: pd.Series | dict,
    shap_summary: pd.DataFrame | None = None,
    optimization_plan: pd.DataFrame | None = None,
) -> WardBrief:
    record = ward_record if isinstance(ward_record, dict) else ward_record.to_dict()
    ward_id = str(_first(record, ("ward_id", "WARD", "ward"), "Citywide"))
    ward_name = str(_first(record, ("ward_name", "WARD", "ward"), f"Ward {ward_id}"))
    lst = _number(_first(record, ("mean_lst", "PREDICTED_LST_C", "LST_C"), None))
    ndvi = _number(record.get("NDVI"))
    ndbi = _number(record.get("NDBI"))
    built = _number(record.get("BUILT_FRACTION"))
    water = _number(record.get("DW_WATER_PROB"))
    driver = _dominant_driver(record, shap_summary)
    heat_phrase = _heat_phrase(lst)
    driver_phrase = _driver_phrase(driver, ndvi, ndbi, built, water)
    intervention = _recommended_intervention(record, optimization_plan)

    summary = f"{ward_name} exhibits {heat_phrase} heat stress driven primarily by {driver_phrase}."
    recommendation = f"{intervention} is expected to provide the strongest near-term marginal benefit for this ward."
    evidence = tuple(
        item
        for item in (
            f"Mean modelled LST: {lst:.2f} C" if lst is not None else None,
            f"NDVI: {ndvi:.2f}" if ndvi is not None else None,
            f"NDBI: {ndbi:.2f}" if ndbi is not None else None,
            f"Built fraction: {built:.2f}" if built is not None else None,
            f"Water probability: {water:.2f}" if water is not None else None,
        )
        if item is not None
    )
    return WardBrief(ward_id=ward_id, ward_name=ward_name, summary=summary, recommendation=recommendation, evidence=evidence)


def generate_intervention_recommendations(
    ward_records: pd.DataFrame,
    shap_summary: pd.DataFrame | None = None,
    optimization_plan: pd.DataFrame | None = None,
    limit: int = 8,
) -> pd.DataFrame:
    if ward_records.empty:
        return pd.DataFrame(columns=["ward_id", "ward_name", "summary", "recommendation", "evidence"])

    sort_column = "hotspot_score" if "hotspot_score" in ward_records.columns else "PREDICTED_LST_C"
    ranked = ward_records.sort_values(sort_column, ascending=False).head(limit) if sort_column in ward_records else ward_records.head(limit)
    briefs = [generate_ward_summary(row._asdict(), shap_summary, optimization_plan) for row in ranked.itertuples(index=False)]
    return pd.DataFrame(
        {
            "ward_id": [brief.ward_id for brief in briefs],
            "ward_name": [brief.ward_name for brief in briefs],
            "summary": [brief.summary for brief in briefs],
            "recommendation": [brief.recommendation for brief in briefs],
            "evidence": ["; ".join(brief.evidence) for brief in briefs],
        }
    )


def generate_city_briefing(
    ward_records: pd.DataFrame,
    shap_values: pd.DataFrame | None = None,
    optimization_plan: pd.DataFrame | None = None,
) -> str:
    if ward_records.empty:
        return "City-level briefing is unavailable until ward heat summaries are loaded."

    ward_count = len(ward_records)
    lst_column = "mean_lst" if "mean_lst" in ward_records.columns else "PREDICTED_LST_C"
    mean_lst = ward_records[lst_column].mean() if lst_column in ward_records else None
    top_ward = _top_ward_name(ward_records)
    shap_summary = _summarize_global_shap(shap_values) if shap_values is not None and not shap_values.empty else pd.DataFrame()
    top_driver = get_driver_label(shap_summary.iloc[0]["feature"]) if not shap_summary.empty else "built form and vegetation indicators"
    plan_phrase = _plan_phrase(optimization_plan)

    heat_sentence = (
        f"Across {ward_count} mapped wards, the mean modelled LST is {mean_lst:.2f} C."
        if mean_lst is not None and pd.notna(mean_lst)
        else f"Across {ward_count} mapped wards, the platform has enough ward-level signal for prioritization."
    )
    return (
        f"{heat_sentence} {top_ward} is the highest-priority ward in the current ranking. "
        f"The leading explanatory signal is {top_driver}. {plan_phrase}"
    )


def _dominant_driver(record: dict, shap_summary: pd.DataFrame | None) -> str:
    if record.get("dominant_driver") and not pd.isna(record.get("dominant_driver")):
        return get_driver_label(str(record["dominant_driver"]))
    if shap_summary is not None and not shap_summary.empty and "feature" in shap_summary:
        return get_driver_label(str(shap_summary.iloc[0]["feature"]))
    return "built-up density and vegetation cover"


def get_driver_label(feature_name: str | None) -> str:
    if not feature_name:
        return "Driver not available"
    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").title())


def _summarize_global_shap(shap_df: pd.DataFrame) -> pd.DataFrame:
    if shap_df.empty:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    reserved = {"base_value", "prediction", "LST_C", "latitude", "longitude", "ward_id", "ward_name"}
    feature_columns = [column for column in shap_df.columns if column not in reserved]
    if not feature_columns:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    return (
        shap_df[feature_columns]
        .abs()
        .mean()
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index(name="mean_abs_shap")
    )


def _driver_phrase(driver: str, ndvi: float | None, ndbi: float | None, built: float | None, water: float | None) -> str:
    low_green = ndvi is not None and ndvi < 0.28
    high_built = (built is not None and built > 0.45) or (ndbi is not None and ndbi > 0.03)
    low_water = water is not None and water < 0.08
    if high_built and low_green:
        return "built-up density and low vegetation cover"
    if high_built:
        return "built-up density"
    if low_green:
        return "low vegetation cover"
    if low_water:
        return "limited blue-space cooling"
    return driver.lower()


def _recommended_intervention(record: dict, optimization_plan: pd.DataFrame | None) -> str:
    ward_id = str(_first(record, ("ward_id", "WARD", "ward"), ""))
    if optimization_plan is not None and not optimization_plan.empty:
        for column in ("ward_id", "ward_name"):
            if column in optimization_plan:
                match = optimization_plan[optimization_plan[column].astype(str).eq(str(record.get(column, ward_id)))]
                if not match.empty and "intervention_type" in match:
                    return str(match.iloc[0]["intervention_type"])
    ndvi = _number(record.get("NDVI"))
    built = _number(record.get("BUILT_FRACTION"))
    water = _number(record.get("DW_WATER_PROB"))
    if ndvi is not None and ndvi < 0.25:
        return "Urban greening"
    if built is not None and built > 0.48:
        return "Cool roofs"
    if water is not None and water < 0.05:
        return "Blue-green infrastructure"
    return "Reflective / high-albedo surfaces"


def _heat_phrase(lst: float | None) -> str:
    if lst is None:
        return "observable"
    if lst >= 39:
        return "severe"
    if lst >= 37:
        return "elevated"
    return "moderate"


def _top_ward_name(ward_records: pd.DataFrame) -> str:
    sort_column = "hotspot_score" if "hotspot_score" in ward_records.columns else "PREDICTED_LST_C"
    if sort_column in ward_records:
        row = ward_records.sort_values(sort_column, ascending=False).iloc[0]
    else:
        row = ward_records.iloc[0]
    name = _first(row.to_dict(), ("ward_name", "WARD", "ward_id"), "The top-ranked ward")
    return f"Ward {name}" if str(name).isdigit() else str(name)


def _plan_phrase(optimization_plan: pd.DataFrame | None) -> str:
    if optimization_plan is None or optimization_plan.empty:
        return "Run the optimization page to translate this diagnosis into budgeted actions."
    impact = optimization_plan["estimated_cooling_c"].sum() if "estimated_cooling_c" in optimization_plan else 0.0
    cost = optimization_plan["estimated_cost_inr"].sum() if "estimated_cost_inr" in optimization_plan else 0.0
    return f"The current optimized package allocates INR {cost:,.0f} and estimates {impact:.2f} C of cumulative ward-scale cooling."


def _first(record: dict, keys: tuple[str, ...], default: object) -> object:
    for key in keys:
        value = record.get(key)
        if value is not None and not pd.isna(value):
            return value
    return default


def _number(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
