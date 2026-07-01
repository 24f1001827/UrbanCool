from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st


POINT_REQUIRED_COLUMNS = {"latitude", "longitude", "LST_C"}
WARD_SUMMARY_COLUMNS = [
    "ward_id",
    "ward_name",
    "mean_lst",
    "hotspot_share",
    "hotspot_score",
    "rank",
    "sample_count",
    "dominant_driver",
    "top_driver_value",
]
WARD_PLANNING_COLUMNS = [
    "PREDICTED_LST_C",
    "NDVI",
    "NDBI",
    "BUILT_FRACTION",
    "DW_WATER_PROB",
    "pixel_count",
    "area_km2",
]
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
WARD_ID_CANDIDATES = ("ward_id", "ward_no", "ward", "wardnum", "id", "objectid", "fid")
WARD_NAME_CANDIDATES = ("ward_name", "name", "ward_label", "wardnam", "ward_no", "ward")
WARD_BOUNDARY_CANDIDATES = (
    Path("data/external/ward_boundaries.geojson"),
    Path("data/external/kmc_wards.geojson"),
    Path("data/external/wards.geojson"),
)
WARD_SUMMARY_CANDIDATES = (
    Path("outputs/ward_heat_summary.geojson"),
    Path("outputs/ward_summary.geojson"),
    Path("outputs/ward_summary.csv"),
    Path("data/processed/ward_summary.csv"),
)
SHAP_RESERVED_COLUMNS = {
    "base_value",
    "prediction",
    "LST_C",
    "latitude",
    "longitude",
    "ward_id",
    "ward_name",
}


@dataclass(frozen=True)
class DashboardArtifacts:
    training_sample_path: Path = Path("data/processed/kolkata_training_sample.csv")
    shap_values_path: Path = Path("outputs/shap_values.csv")
    ward_boundaries_path: Path = Path("data/external/ward_boundaries.geojson")
    ward_summary_path: Path = Path("outputs/ward_summary.geojson")


def _resolve_existing_path(primary: Path, fallbacks: tuple[Path, ...]) -> Path:
    for candidate in (primary, *fallbacks):
        if candidate.exists():
            return candidate
    return primary


def _empty_points_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["latitude", "longitude", "LST_C"])


def _empty_ward_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=WARD_SUMMARY_COLUMNS)


def _empty_wards_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"ward_id": [], "ward_name": []}, geometry=[], crs="EPSG:4326")


@st.cache_data(show_spinner=False)
def load_training_points(artifacts: DashboardArtifacts = DashboardArtifacts()) -> pd.DataFrame:
    path = artifacts.training_sample_path
    if not path.exists():
        return _empty_points_frame()

    data = pd.read_csv(path)
    if not POINT_REQUIRED_COLUMNS.issubset(data.columns):
        return _empty_points_frame()

    return data.dropna(subset=["latitude", "longitude", "LST_C"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_shap_values(artifacts: DashboardArtifacts = DashboardArtifacts()) -> pd.DataFrame:
    path = artifacts.shap_values_path
    if not path.exists():
        return pd.DataFrame()

    data = pd.read_csv(path)
    if data.empty:
        return data

    return data.reset_index(drop=True)


def _normalize_ward_columns(wards: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if wards.empty:
        return _empty_wards_gdf()

    normalized = wards.copy()
    columns_lower = {column.lower(): column for column in normalized.columns}

    ward_id_source = next((columns_lower[key] for key in WARD_ID_CANDIDATES if key in columns_lower), None)
    ward_name_source = next((columns_lower[key] for key in WARD_NAME_CANDIDATES if key in columns_lower), None)

    if ward_id_source is None:
        normalized["ward_id"] = normalized.index.astype(str)
    else:
        normalized["ward_id"] = normalized[ward_id_source].astype(str)

    if ward_name_source is None:
        normalized["ward_name"] = "Ward " + normalized["ward_id"].astype(str)
    else:
        normalized["ward_name"] = normalized[ward_name_source].astype(str)

    if normalized.crs is None:
        normalized = normalized.set_crs("EPSG:4326")
    else:
        normalized = normalized.to_crs("EPSG:4326")

    return normalized[["ward_id", "ward_name", "geometry"]].copy()


@st.cache_data(show_spinner=False)
def load_ward_boundaries(artifacts: DashboardArtifacts = DashboardArtifacts()) -> gpd.GeoDataFrame:
    path = _resolve_existing_path(
        artifacts.ward_boundaries_path,
        tuple(candidate for candidate in WARD_BOUNDARY_CANDIDATES if candidate != artifacts.ward_boundaries_path),
    )
    if not path.exists():
        return _empty_wards_gdf()

    wards = gpd.read_file(path)
    return _normalize_ward_columns(wards)


@st.cache_data(show_spinner=False)
def load_ward_summaries(artifacts: DashboardArtifacts = DashboardArtifacts()) -> pd.DataFrame:
    default_artifacts = DashboardArtifacts()
    if artifacts.ward_summary_path == default_artifacts.ward_summary_path:
        path = _resolve_existing_path(
            artifacts.ward_summary_path,
            tuple(candidate for candidate in WARD_SUMMARY_CANDIDATES if candidate != artifacts.ward_summary_path),
        )
    else:
        path = artifacts.ward_summary_path
    if not path.exists():
        return _empty_ward_summary_frame()

    if path.suffix.lower() in {".geojson", ".json"}:
        summary_gdf = gpd.read_file(path)
        if summary_gdf.empty:
            return _empty_ward_summary_frame()
        summary = _normalize_precomputed_ward_summary(summary_gdf)
        if "geometry" in summary_gdf:
            projected = summary_gdf.to_crs("EPSG:32645") if summary_gdf.crs is not None else summary_gdf.set_crs("EPSG:4326").to_crs("EPSG:32645")
            summary["area_km2"] = projected.geometry.area / 1_000_000.0
        if summary.empty:
            return _empty_ward_summary_frame()
        summary = pd.DataFrame(summary.drop(columns="geometry", errors="ignore"))
    else:
        summary = pd.read_csv(path)
        summary = _normalize_precomputed_ward_summary(summary)

    for column in [*WARD_SUMMARY_COLUMNS, *WARD_PLANNING_COLUMNS]:
        if column not in summary.columns:
            summary[column] = np.nan

    return summary[[*WARD_SUMMARY_COLUMNS, *WARD_PLANNING_COLUMNS]].copy()


def _normalize_precomputed_ward_summary(summary: pd.DataFrame) -> pd.DataFrame:
    normalized = summary.copy()
    columns_lower = {column.lower(): column for column in normalized.columns}

    ward_id_source = next((columns_lower[key] for key in WARD_ID_CANDIDATES if key in columns_lower), None)
    ward_name_source = next((columns_lower[key] for key in WARD_NAME_CANDIDATES if key in columns_lower), None)
    if ward_id_source is not None and "ward_id" not in normalized:
        normalized["ward_id"] = normalized[ward_id_source].astype(str)
    if ward_name_source is not None and "ward_name" not in normalized:
        normalized["ward_name"] = "Ward " + normalized[ward_name_source].astype(str)
    if "ward_id" not in normalized:
        normalized["ward_id"] = normalized.index.astype(str)
    if "ward_name" not in normalized:
        normalized["ward_name"] = "Ward " + normalized["ward_id"].astype(str)

    if "PREDICTED_LST_C" in normalized and "mean_lst" not in normalized:
        normalized["mean_lst"] = normalized["PREDICTED_LST_C"]
    if "mean_lst" in normalized and "PREDICTED_LST_C" not in normalized:
        normalized["PREDICTED_LST_C"] = normalized["mean_lst"]
    if "pixel_count" in normalized and "sample_count" not in normalized:
        normalized["sample_count"] = normalized["pixel_count"]

    if "hotspot_share" not in normalized:
        lst = pd.to_numeric(normalized.get("mean_lst"), errors="coerce")
        threshold = lst.quantile(0.67) if lst.notna().any() else np.nan
        normalized["hotspot_share"] = np.where(lst.notna(), (lst >= threshold).astype(float), np.nan)
    if "hotspot_score" not in normalized:
        lst = pd.to_numeric(normalized.get("mean_lst"), errors="coerce")
        share = pd.to_numeric(normalized.get("hotspot_share"), errors="coerce").fillna(0.0)
        normalized["hotspot_score"] = lst * 0.6 + share * 100.0 * 0.4
    if "rank" not in normalized:
        normalized["rank"] = pd.to_numeric(normalized["hotspot_score"], errors="coerce").rank(ascending=False, method="dense")

    return normalized


def derive_hotspot_metrics(points_df: pd.DataFrame) -> pd.DataFrame:
    if points_df.empty:
        return points_df.assign(
            hotspot_zscore=pd.Series(dtype=float),
            hotspot_score=pd.Series(dtype=float),
            hotspot_label=pd.Series(dtype=str),
            is_hotspot=pd.Series(dtype=bool),
        )

    data = points_df.copy()
    mean_lst = float(data["LST_C"].mean())
    std_lst = float(data["LST_C"].std(ddof=0))

    if std_lst <= 0:
        zscore = np.zeros(len(data))
    else:
        zscore = (data["LST_C"] - mean_lst) / std_lst

    hotspot_score = np.clip(((zscore + 2.0) / 4.0) * 100.0, 0.0, 100.0)

    labels = np.select(
        [zscore >= 1.5, zscore >= 0.75, zscore >= 0.0],
        ["Severe", "Moderate", "Watch"],
        default="Background",
    )

    data["hotspot_zscore"] = zscore
    data["hotspot_score"] = hotspot_score
    data["hotspot_label"] = labels
    data["is_hotspot"] = data["hotspot_label"].ne("Background")
    return data


def _points_to_geodataframe(points_df: pd.DataFrame) -> gpd.GeoDataFrame:
    points = points_df.copy()
    points["point_index"] = np.arange(len(points))
    return gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy(points["longitude"], points["latitude"]),
        crs="EPSG:4326",
    )


def _get_shap_feature_columns(shap_df: pd.DataFrame) -> list[str]:
    return [column for column in shap_df.columns if column not in SHAP_RESERVED_COLUMNS]


def merge_shap_with_points(points_df: pd.DataFrame, shap_df: pd.DataFrame) -> pd.DataFrame:
    if points_df.empty or shap_df.empty:
        return pd.DataFrame()

    points = points_df.reset_index(drop=True).copy()
    shap_data = shap_df.reset_index(drop=True).copy()

    if {"latitude", "longitude"}.issubset(shap_data.columns):
        merged = points.merge(
            shap_data,
            on=["latitude", "longitude"],
            how="inner",
            suffixes=("", "_shap"),
        )
        return merged

    if len(points) != len(shap_data):
        return pd.DataFrame()

    points["row_id"] = np.arange(len(points))
    shap_data["row_id"] = np.arange(len(shap_data))
    merged = points.merge(shap_data, on="row_id", how="inner", suffixes=("", "_shap"))
    return merged.drop(columns=["row_id"], errors="ignore")


def derive_ward_summary(
    points_df: pd.DataFrame,
    wards_gdf: gpd.GeoDataFrame,
    shap_df: pd.DataFrame | None = None,
) -> gpd.GeoDataFrame:
    if points_df.empty or wards_gdf.empty:
        return _empty_wards_gdf()

    hotspot_points = derive_hotspot_metrics(points_df)
    points_gdf = _points_to_geodataframe(hotspot_points)
    wards = _normalize_ward_columns(wards_gdf)
    joined = gpd.sjoin(points_gdf, wards, how="inner", predicate="within")

    if joined.empty:
        return wards.assign(
            mean_lst=np.nan,
            hotspot_share=np.nan,
            hotspot_score=np.nan,
            rank=np.nan,
            sample_count=0,
            dominant_driver=pd.NA,
            top_driver_value=np.nan,
        )

    grouped = (
        joined.groupby(["ward_id", "ward_name"], dropna=False)
        .agg(
            mean_lst=("LST_C", "mean"),
            hotspot_share=("is_hotspot", "mean"),
            sample_count=("point_index", "count"),
        )
        .reset_index()
    )
    grouped["hotspot_score"] = grouped["mean_lst"] * 0.6 + grouped["hotspot_share"] * 100.0 * 0.4
    grouped["rank"] = grouped["hotspot_score"].rank(ascending=False, method="dense").astype(int)

    if shap_df is not None and not shap_df.empty:
        merged_shap = merge_shap_with_points(joined.drop(columns="geometry"), shap_df)
        feature_columns = [column for column in _get_shap_feature_columns(shap_df) if column in merged_shap.columns]
        if not merged_shap.empty and feature_columns:
            shap_long = (
                merged_shap[["ward_id", *feature_columns]]
                .melt(id_vars="ward_id", var_name="driver", value_name="shap_value")
                .dropna(subset=["shap_value"])
            )
            if not shap_long.empty:
                driver_summary = (
                    shap_long.assign(abs_shap=lambda df: df["shap_value"].abs())
                    .groupby(["ward_id", "driver"], as_index=False)["abs_shap"]
                    .mean()
                    .sort_values(["ward_id", "abs_shap"], ascending=[True, False])
                    .drop_duplicates("ward_id")
                    .rename(columns={"driver": "dominant_driver", "abs_shap": "top_driver_value"})
                )
                grouped = grouped.merge(driver_summary, on="ward_id", how="left")

    if "dominant_driver" not in grouped.columns:
        grouped["dominant_driver"] = pd.NA
    if "top_driver_value" not in grouped.columns:
        grouped["top_driver_value"] = np.nan

    ward_summary = wards.merge(grouped, on=["ward_id", "ward_name"], how="left")
    ward_summary["sample_count"] = ward_summary["sample_count"].fillna(0).astype(int)
    return ward_summary


def summarize_global_shap(shap_df: pd.DataFrame) -> pd.DataFrame:
    if shap_df.empty:
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "feature_label"])

    feature_columns = _get_shap_feature_columns(shap_df)
    if not feature_columns:
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "feature_label"])

    summary = (
        shap_df[feature_columns]
        .abs()
        .mean()
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index(name="mean_abs_shap")
    )
    summary["feature_label"] = summary["feature"].map(FEATURE_LABELS).fillna(summary["feature"])
    return summary


def get_driver_label(feature_name: str | None) -> str:
    if not feature_name:
        return "Driver not available"
    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").title())


def get_demo_status(artifacts: DashboardArtifacts = DashboardArtifacts()) -> dict[str, Any]:
    points = load_training_points(artifacts)
    shap_df = load_shap_values(artifacts)
    wards = load_ward_boundaries(artifacts)
    ward_summary = load_ward_summaries(artifacts)

    return {
        "training_points_ready": not points.empty,
        "training_point_count": int(len(points)),
        "shap_ready": not shap_df.empty,
        "shap_row_count": int(len(shap_df)),
        "ward_boundaries_ready": not wards.empty,
        "ward_count": int(len(wards)),
        "ward_summary_ready": not ward_summary.empty,
        "ward_summary_count": int(len(ward_summary)),
        "artifacts": artifacts,
    }
