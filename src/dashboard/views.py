from __future__ import annotations

from typing import Iterable

import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.dashboard.components import render_dataframe_preview, render_empty_state, render_status_pill
from src.dashboard.data import (
    DashboardArtifacts,
    derive_hotspot_metrics,
    derive_ward_summary,
    get_demo_status,
    get_driver_label,
    load_shap_values,
    load_training_points,
    load_ward_boundaries,
    load_ward_summaries,
    merge_shap_with_points,
    summarize_global_shap,
)


PAGE_ORDER = [
    "Overview",
    "Hotspots Map",
    "Ward Rankings",
    "Driver Explanations",
    "Scenarios",
]

HOTSPOT_COLORS = {
    "Background": "#94a3b8",
    "Watch": "#facc15",
    "Moderate": "#f97316",
    "Severe": "#dc2626",
}


def render_app(artifacts: DashboardArtifacts = DashboardArtifacts()) -> None:
    st.sidebar.title("UrbanCool AI")
    st.sidebar.caption("Artifact-first hackathon dashboard")
    page = st.sidebar.radio("Navigate", PAGE_ORDER)
    st.sidebar.divider()
    _render_sidebar_status(artifacts)

    if page == "Overview":
        render_overview_page(artifacts)
    elif page == "Hotspots Map":
        render_hotspots_page(artifacts)
    elif page == "Ward Rankings":
        render_ward_rankings_page(artifacts)
    elif page == "Driver Explanations":
        render_driver_explanations_page(artifacts)
    else:
        render_scenarios_page(artifacts)


def _render_sidebar_status(artifacts: DashboardArtifacts) -> None:
    status = get_demo_status(artifacts)
    render_status_pill(
        "Training sample",
        status["training_points_ready"],
        f"{status['training_point_count']:,} points available" if status["training_points_ready"] else "Expected at data/processed/kolkata_training_sample.csv",
    )
    render_status_pill(
        "SHAP outputs",
        status["shap_ready"],
        f"{status['shap_row_count']:,} explanation rows available" if status["shap_ready"] else "Expected at outputs/shap_values.csv",
    )
    render_status_pill(
        "Ward boundaries",
        status["ward_boundaries_ready"],
        f"{status['ward_count']:,} ward polygons available" if status["ward_boundaries_ready"] else "Add a local ward GeoJSON artifact to enable rankings",
    )


def render_overview_page(artifacts: DashboardArtifacts) -> None:
    st.title("UrbanCool AI Dashboard")
    st.caption("Heat stress mapping and intervention planning for a hackathon demo workflow.")

    status = get_demo_status(artifacts)
    points = derive_hotspot_metrics(load_training_points(artifacts))

    st.subheader("Demo Readiness")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Sampled Locations", f"{status['training_point_count']:,}")
    mean_lst = f"{points['LST_C'].mean():.2f} C" if not points.empty else "N/A"
    metric_cols[1].metric("Mean LST", mean_lst)
    metric_cols[2].metric("Ward Boundaries", f"{status['ward_count']:,}" if status["ward_boundaries_ready"] else "Pending")
    metric_cols[3].metric("SHAP Rows", f"{status['shap_row_count']:,}" if status["shap_ready"] else "Not generated")

    if points.empty:
        render_empty_state(
            "Generate the training sample first",
            "The dashboard is ready, but it needs a sampled training artifact before the heat views can render.",
            [
                "Run `python -m src.data_pipeline.gee_fetch --export-csv data/processed/kolkata_training_sample.csv --sample-size 100000` from the project environment.",
                "Return here and refresh the app to unlock the map and hotspot previews.",
            ],
        )
        return

    st.subheader("At a Glance")
    summary_cols = st.columns((1.3, 1))
    with summary_cols[0]:
        st.map(points[["latitude", "longitude"]].dropna().head(2000), use_container_width=True)
    with summary_cols[1]:
        hotspot_counts = points["hotspot_label"].value_counts().reindex(
            ["Severe", "Moderate", "Watch", "Background"], fill_value=0
        )
        st.markdown("**Hotspot mix**")
        st.bar_chart(hotspot_counts)
        st.markdown("**Top hotspot sample points**")
        hotspot_preview = points.sort_values("hotspot_score", ascending=False)[
            ["latitude", "longitude", "LST_C", "hotspot_label", "hotspot_score"]
        ]
        st.dataframe(hotspot_preview.head(8), use_container_width=True, hide_index=True)

    st.subheader("Artifact Status")
    readiness_cols = st.columns(3)
    with readiness_cols[0]:
        render_status_pill(
            "Heat hotspot visualization",
            True,
            "Ready from the sampled training artifact.",
        )
    with readiness_cols[1]:
        render_status_pill(
            "Ward rankings",
            status["ward_boundaries_ready"],
            "Will auto-enable once a local ward boundary GeoJSON is present.",
        )
    with readiness_cols[2]:
        render_status_pill(
            "Driver explanations",
            status["shap_ready"],
            "Global explanations unlock when SHAP outputs are available.",
        )


def render_hotspots_page(artifacts: DashboardArtifacts) -> None:
    st.title("Heat Hotspots Map")
    st.caption("Point-based hotspot visualization derived from the sampled training artifact.")
    points = derive_hotspot_metrics(load_training_points(artifacts))

    if points.empty:
        render_empty_state(
            "Hotspot map unavailable",
            "No sampled training artifact was found, so the hotspot map cannot render yet.",
            [
                "Generate `data/processed/kolkata_training_sample.csv` from the existing pipeline.",
                "Reload the app. No live Earth Engine integration is required for this dashboard.",
            ],
        )
        return

    severity_options = ["All", "Severe", "Moderate", "Watch", "Background"]
    control_cols = st.columns((1, 1.4, 1))
    severity = control_cols[0].selectbox("Severity", severity_options)
    lst_range = control_cols[1].slider(
        "LST range (C)",
        min_value=float(points["LST_C"].min()),
        max_value=float(points["LST_C"].max()),
        value=(float(points["LST_C"].min()), float(points["LST_C"].max())),
    )
    max_points = control_cols[2].slider("Max points displayed", min_value=100, max_value=min(5000, len(points)), value=min(1500, len(points)), step=100)

    filtered = points[(points["LST_C"] >= lst_range[0]) & (points["LST_C"] <= lst_range[1])]
    if severity != "All":
        filtered = filtered[filtered["hotspot_label"] == severity]
    filtered = filtered.sort_values("hotspot_score", ascending=False).head(max_points)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Displayed points", f"{len(filtered):,}")
    summary_cols[1].metric("Mean displayed LST", f"{filtered['LST_C'].mean():.2f} C" if not filtered.empty else "N/A")
    summary_cols[2].metric("Hotspot share", f"{filtered['is_hotspot'].mean() * 100:.1f}%" if not filtered.empty else "N/A")
    summary_cols[3].metric("Peak hotspot score", f"{filtered['hotspot_score'].max():.1f}" if not filtered.empty else "N/A")

    if filtered.empty:
        st.warning("No points matched the current filter combination.")
        return

    st_folium(_build_hotspot_map(filtered), use_container_width=True, height=560)
    render_dataframe_preview(
        filtered[["latitude", "longitude", "LST_C", "hotspot_label", "hotspot_score"]],
        "Filtered Hotspot Sample",
        limit=15,
    )


def _build_hotspot_map(points: pd.DataFrame) -> folium.Map:
    center = [points["latitude"].mean(), points["longitude"].mean()]
    hotspot_map = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
    cluster = MarkerCluster().add_to(hotspot_map)

    for row in points.itertuples(index=False):
        color = HOTSPOT_COLORS.get(row.hotspot_label, HOTSPOT_COLORS["Background"])
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=1,
            popup=(
                f"LST: {row.LST_C:.2f} C<br>"
                f"Severity: {row.hotspot_label}<br>"
                f"Hotspot score: {row.hotspot_score:.1f}"
            ),
        ).add_to(cluster)

    return hotspot_map


def render_ward_rankings_page(artifacts: DashboardArtifacts) -> None:
    st.title("Ward Rankings")
    st.caption("Ward-level prioritization from local artifacts, with in-memory aggregation when possible.")

    points = load_training_points(artifacts)
    wards = load_ward_boundaries(artifacts)
    precomputed = load_ward_summaries(artifacts)

    if not precomputed.empty and wards.empty:
        st.warning("A ward summary artifact exists, but no ward boundary GeoJSON is available for mapping.")
        st.dataframe(precomputed.sort_values("rank"), use_container_width=True, hide_index=True)
        return

    if wards.empty:
        render_empty_state(
            "Ward rankings pending",
            "Add a local ward boundary GeoJSON artifact to enable choropleths and ward ranking tables.",
            [
                "Place a ward file at `data/external/ward_boundaries.geojson`, `data/external/kmc_wards.geojson`, or `data/external/wards.geojson`.",
                "Keep one stable ward identifier and one display name field; the loader will normalize them to `ward_id` and `ward_name`.",
            ],
        )
        return

    ward_summary = derive_ward_summary(points, wards) if precomputed.empty else wards.merge(precomputed, on=["ward_id", "ward_name"], how="left")
    if ward_summary.empty or ward_summary["sample_count"].fillna(0).sum() == 0:
        st.warning("Ward polygons are available, but no points could be assigned to them yet.")
        return

    ranked = (
        ward_summary.drop(columns="geometry")
        .sort_values(["rank", "hotspot_score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    selected_ward_name = st.selectbox("Inspect ward", ranked["ward_name"].dropna().tolist())
    selected = ranked[ranked["ward_name"] == selected_ward_name].iloc[0]

    detail_cols = st.columns(4)
    detail_cols[0].metric("Rank", int(selected["rank"]) if pd.notna(selected["rank"]) else "N/A")
    detail_cols[1].metric("Mean LST", f"{selected['mean_lst']:.2f} C" if pd.notna(selected["mean_lst"]) else "N/A")
    detail_cols[2].metric("Hotspot share", f"{selected['hotspot_share'] * 100:.1f}%" if pd.notna(selected["hotspot_share"]) else "N/A")
    detail_cols[3].metric("Sample points", f"{int(selected['sample_count']):,}")

    map_col, table_col = st.columns((1.2, 1))
    with map_col:
        st_folium(_build_ward_map(ward_summary, selected["ward_id"]), use_container_width=True, height=560)
    with table_col:
        st.markdown("**Ward ranking table**")
        display = ranked[["rank", "ward_name", "mean_lst", "hotspot_share", "hotspot_score", "sample_count"]].copy()
        display["hotspot_share"] = (display["hotspot_share"] * 100).round(1)
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.caption("Ranking formula: `hotspot_score = 0.6 * mean_lst + 0.4 * hotspot_share * 100`.")


def _build_ward_map(ward_summary: gpd.GeoDataFrame, selected_ward_id: str) -> folium.Map:
    centroid = ward_summary.geometry.unary_union.centroid
    ward_map = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles="CartoDB positron")

    def style_function(feature: dict) -> dict:
        ward_id = str(feature["properties"].get("ward_id"))
        is_selected = ward_id == str(selected_ward_id)
        score = feature["properties"].get("hotspot_score")
        fill_color = "#ef4444" if pd.notna(score) and score >= ward_summary["hotspot_score"].median() else "#fca5a5"
        if is_selected:
            fill_color = "#2563eb"
        return {
            "fillColor": fill_color,
            "color": "#1f2937",
            "weight": 2 if is_selected else 1,
            "fillOpacity": 0.55,
        }

    tooltip = folium.features.GeoJsonTooltip(
        fields=["ward_name", "rank", "mean_lst", "hotspot_share", "sample_count"],
        aliases=["Ward", "Rank", "Mean LST", "Hotspot share", "Sample count"],
        localize=True,
    )

    folium.GeoJson(ward_summary.to_json(), style_function=style_function, tooltip=tooltip).add_to(ward_map)
    return ward_map


def render_driver_explanations_page(artifacts: DashboardArtifacts) -> None:
    st.title("Driver Explanations")
    st.caption("SHAP-based explanations for global drivers, with ward-level detail when artifacts line up.")

    points = load_training_points(artifacts)
    shap_df = load_shap_values(artifacts)
    wards = load_ward_boundaries(artifacts)

    if shap_df.empty:
        render_empty_state(
            "Driver explanations unavailable",
            "No SHAP artifact was found yet, so explanation panels cannot be rendered.",
            [
                "Train the baseline model and run `python -m src.explainability.shap_baseline`.",
                "Ensure `outputs/shap_values.csv` exists, then refresh the dashboard.",
            ],
        )
        return

    summary = summarize_global_shap(shap_df)
    top_cols = st.columns((1, 1))
    with top_cols[0]:
        st.markdown("**Global driver importance**")
        st.bar_chart(summary.set_index("feature_label")["mean_abs_shap"].head(12))
    with top_cols[1]:
        st.markdown("**Top global drivers**")
        st.dataframe(summary.head(12), use_container_width=True, hide_index=True)

    merged = merge_shap_with_points(points, shap_df)
    ward_summary = derive_ward_summary(points, wards, shap_df) if not wards.empty and not merged.empty else gpd.GeoDataFrame()

    st.subheader("Why is this area hot?")
    if not ward_summary.empty and ward_summary["dominant_driver"].notna().any():
        options = ward_summary["ward_name"].dropna().tolist()
        ward_name = st.selectbox("Explain ward", options)
        ward_row = ward_summary[ward_summary["ward_name"] == ward_name].iloc[0]
        dominant_label = get_driver_label(ward_row.get("dominant_driver"))
        st.success(
            (
                f"**{ward_name}** is currently led by **{dominant_label}** in the SHAP signal. "
                f"The ward ranks **#{int(ward_row['rank'])}** with a mean LST of "
                f"**{ward_row['mean_lst']:.2f} C** and a hotspot share of "
                f"**{ward_row['hotspot_share'] * 100:.1f}%**."
            )
        )
    else:
        top_feature = summary.iloc[0] if not summary.empty else None
        label = top_feature["feature_label"] if top_feature is not None else "driver signals"
        st.info(
            (
                "Ward-level SHAP aggregation is not available from the current artifacts, "
                f"so the dashboard is showing the global explanation view. The strongest current driver is **{label}**."
            )
        )


def render_scenarios_page(artifacts: DashboardArtifacts) -> None:
    st.title("Scenario Simulation")
    st.caption("UI placeholder for the next milestone. No cooling estimates are fabricated here.")

    wards = load_ward_boundaries(artifacts)
    ward_options: list[str] = ["Citywide"]
    if not wards.empty:
        ward_options.extend(wards["ward_name"].dropna().tolist())

    control_cols = st.columns(3)
    intervention = control_cols[0].selectbox(
        "Intervention type",
        ["Tree cover", "Cool roofs", "Albedo change", "Blue-green infrastructure"],
    )
    target = control_cols[1].selectbox("Target geography", ward_options)
    intensity = control_cols[2].slider("Intervention intensity", min_value=10, max_value=100, value=40, step=5)

    st.button("Simulate scenario", disabled=True, use_container_width=False)
    st.warning("Scenario engine not implemented yet. This page is intentionally a polished placeholder.")

    st.markdown("**Planned future outputs**")
    st.markdown(
        "\n".join(
            [
                "- Estimated cooling impact in degrees Celsius",
                "- Suitability score for the selected intervention and geography",
                "- Recommended spatial placement tied to hotspot and ward context",
                "- Links back to trained model artifacts and ward summaries",
            ]
        )
    )

    st.markdown("**Current placeholder input contract**")
    st.json(
        {
            "intervention_type": intervention,
            "target_geography": target,
            "intensity": intensity,
            "status": "placeholder_only",
        }
    )
