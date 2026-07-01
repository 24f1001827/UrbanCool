from __future__ import annotations

from math import isnan

import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.dashboard.components import (
    inject_global_styles,
    open_panel,
    render_dataframe_preview,
    render_empty_state,
    render_hero_banner,
    render_insight_card,
    render_kpi_card,
    render_legend,
    render_presentation_band,
    render_section_header,
    render_sidebar_brand,
    render_status_pill,
    render_story_panel,
)
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
from src.llm_layer import (
    generate_city_briefing,
    generate_intervention_recommendations,
    generate_ward_summary,
)
from src.optimization import DEFAULT_INTERVENTION_COSTS_INR_PER_KM2, GreedyCoolingOptimizer, optimize_cooling_plan
from src.scenario_engine import SUPPORTED_INTERVENTIONS, simulate_for_wards, simulate_intervention


PAGE_ORDER = [
    "Overview",
    "Hotspots Map",
    "Ward Rankings",
    "Driver Explanations",
    "Scenario Simulation",
    "Optimization",
    "AI Recommendations",
]

HOTSPOT_COLORS = {
    "Background": "#cbd5e1",
    "Watch": "#facc15",
    "Moderate": "#f97316",
    "Severe": "#dc2626",
}

STORY_STEPS = [
    ("Identify hotspot", "Start with the map to locate where urban heat stress concentrates across the sampled surface."),
    ("Understand drivers", "Use SHAP-backed insights to see whether built-up density, vegetation loss, or radiation is dominating."),
    ("Compare wards", "Shift to ward rankings to prioritize which areas need intervention first."),
    ("Evaluate interventions", "Use the scenario view as the handoff point for future cooling strategy simulation."),
]


def render_app(artifacts: DashboardArtifacts = DashboardArtifacts()) -> None:
    inject_global_styles()
    with st.sidebar:
        render_sidebar_brand()
        st.markdown("<div class='uc-sidebar-section'>Platform Modules</div>", unsafe_allow_html=True)
        page = st.radio("Platform Modules", PAGE_ORDER, label_visibility="collapsed")
        st.divider()
        st.markdown("<div class='uc-sidebar-section'>Data Readiness</div>", unsafe_allow_html=True)
        _render_sidebar_status(artifacts)

    if page == "Overview":
        render_overview_page(artifacts)
    elif page == "Hotspots Map":
        render_hotspots_page(artifacts)
    elif page == "Ward Rankings":
        render_ward_rankings_page(artifacts)
    elif page == "Driver Explanations":
        render_driver_explanations_page(artifacts)
    elif page == "Scenario Simulation":
        render_scenarios_page(artifacts)
    elif page == "Optimization":
        render_optimization_page(artifacts)
    else:
        render_ai_recommendations_page(artifacts)


def _render_sidebar_status(artifacts: DashboardArtifacts) -> None:
    status = get_demo_status(artifacts)
    render_status_pill(
        "Training sample",
        status["training_points_ready"],
        f"{status['training_point_count']:,} points loaded" if status["training_points_ready"] else "Expected at data/processed/kolkata_training_sample.csv",
        compact=True,
    )
    render_status_pill(
        "SHAP outputs",
        status["shap_ready"],
        f"{status['shap_row_count']:,} explanation rows ready" if status["shap_ready"] else "Expected at outputs/shap_values.csv",
        compact=True,
    )
    render_status_pill(
        "Ward boundaries",
        status["ward_boundaries_ready"],
        f"{status['ward_count']:,} polygons ready" if status["ward_boundaries_ready"] else "Add a local ward GeoJSON to unlock rankings",
        compact=True,
    )


def render_overview_page(artifacts: DashboardArtifacts) -> None:
    status = get_demo_status(artifacts)
    points = derive_hotspot_metrics(load_training_points(artifacts))
    wards = load_ward_boundaries(artifacts)
    ward_summary = derive_ward_summary(points, wards) if not points.empty and not wards.empty else gpd.GeoDataFrame()
    hotspot_count = int(points["is_hotspot"].sum()) if not points.empty else 0
    high_risk_wards = int((ward_summary["rank"] <= 5).sum()) if not ward_summary.empty and "rank" in ward_summary else 0
    dominant_driver = _resolve_overview_driver(artifacts, points, wards)
    coverage = _format_coverage(status)

    render_hero_banner(
        "UrbanCool AI<br>Urban Heat Intelligence Platform",
        "Identify hotspots • Explain drivers • Prioritize interventions",
        "Hackathon Demo Mode",
        [
            "Artifact-first intelligence",
            "Geospatial hotspot discovery",
            "Explainable urban heat diagnostics",
        ],
    )

    hero_cols = st.columns((1.55, 0.95))
    with hero_cols[0]:
        render_presentation_band(
            [
                ("What it does", "Transforms sampled geospatial heat signals into an actionable intelligence surface."),
                ("Where it starts", "A map-first workflow makes hotspot geography visible in the first glance."),
                ("Why it matters", "Driver explanations connect elevated heat to built form, vegetation, and urban morphology."),
                ("What comes next", "Ward ranking and intervention planning frame the path to cooling strategy decisions."),
            ]
        )
    with hero_cols[1]:
        render_story_panel(STORY_STEPS)

    _render_readiness_strip(status)

    if points.empty:
        render_empty_state(
            "Platform ready, artifacts missing",
            "The premium dashboard shell is live, but it needs the sampled training artifact before the intelligence surface can render.",
            [
                "Run `python -m src.data_pipeline.gee_fetch --export-csv data/processed/kolkata_training_sample.csv --sample-size 100000` from the project environment.",
                "Refresh the app to unlock the map, KPI cards, and hotspot workflow.",
            ],
        )
        return

    overview_cols = st.columns((1.42, 0.88))
    with overview_cols[0]:
        render_section_header(
            "Heat Overview",
            "The map appears first so judges can immediately see where urban heat stress is clustering.",
        )
        st_folium(_build_hotspot_map(points.sort_values("hotspot_score", ascending=False).head(1200), radius=4.5), use_container_width=True, height=470)
        render_legend(
            [
                ("Background", HOTSPOT_COLORS["Background"]),
                ("Watch", HOTSPOT_COLORS["Watch"]),
                ("Moderate", HOTSPOT_COLORS["Moderate"]),
                ("Severe", HOTSPOT_COLORS["Severe"]),
            ]
        )
    with overview_cols[1]:
        render_section_header(
            "Executive Signal",
            "The first ten seconds should answer what the platform does, where the heat is, and what should be investigated next.",
        )
        render_insight_card(
            "Hotspot overview",
            _build_hotspot_overview_copy(points),
            tone="orange",
        )
        render_insight_card(
            "Driver overview",
            f"{dominant_driver} is currently the strongest explanation signal available from the loaded artifacts.",
            tone="blue",
        )
        render_insight_card(
            "Action framing",
            "Use the Hotspots Map to inspect candidate zones, then move to Ward Rankings to identify which areas deserve the first intervention budget.",
            tone="green",
        )

    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        render_kpi_card("Mean Land Surface Temperature", _format_value(points["LST_C"].mean(), suffix=" C"), "Representative average from the currently loaded sample.", accent="linear-gradient(90deg, #2563eb, #0f9f6e)")
    with kpi_cols[1]:
        render_kpi_card("Hotspot Count", f"{hotspot_count:,}", "Sample points currently classified above background conditions.", accent="linear-gradient(90deg, #f97316, #dc2626)")
    with kpi_cols[2]:
        render_kpi_card("High Risk Wards", f"{high_risk_wards:,}" if wards is not None is not False else "0", "Top-ranked wards available once geometry is loaded.", accent="linear-gradient(90deg, #0f9f6e, #2563eb)")
    with kpi_cols[3]:
        render_kpi_card("Dominant Driver", dominant_driver, "Highest-priority explanatory signal in the current artifact set.", accent="linear-gradient(90deg, #2563eb, #7c3aed)")
    with kpi_cols[4]:
        render_kpi_card("Model Coverage", coverage, "Coverage across points, SHAP outputs, and ward geometry.", accent="linear-gradient(90deg, #0f9f6e, #f97316)")

    insight_cols = st.columns((1.05, 0.95))
    with insight_cols[0]:
        hotspot_preview = points.sort_values("hotspot_score", ascending=False)[
            ["latitude", "longitude", "LST_C", "hotspot_label", "hotspot_score"]
        ]
        render_dataframe_preview(
            hotspot_preview,
            "Priority Hotspot Sample",
            limit=8,
            caption="These are the highest-intensity sample points in the current artifact set.",
        )
    with insight_cols[1]:
        render_section_header(
            "Readiness and Next Actions",
            "The dashboard stays honest about what is loaded, what is inferred, and which module judges should inspect next.",
        )
        render_insight_card(
            "Judge walkthrough",
            "Open with the map, call out the hotspot clusters, highlight the leading driver signal, then transition into ward prioritization for intervention planning.",
            tone="blue",
        )
        render_insight_card(
            "Artifact health",
            _build_artifact_health_copy(status),
            tone="green" if status["training_points_ready"] and status["shap_ready"] else "orange",
        )


def render_hotspots_page(artifacts: DashboardArtifacts) -> None:
    points = derive_hotspot_metrics(load_training_points(artifacts))
    render_hero_banner(
        "Heat Hotspots Map",
        "Map-first spatial exploration for sampled land-surface heat intensity.",
        "Interactive Analysis",
        ["Heat palette", "Filterable hotspot classes", "Presentation-ready geospatial canvas"],
    )

    if points.empty:
        render_empty_state(
            "Hotspot map unavailable",
            "The map surface is ready, but the sampled training artifact is missing.",
            [
                "Generate `data/processed/kolkata_training_sample.csv` from the existing pipeline.",
                "Reload the app. No live Earth Engine integration is required for this experience.",
            ],
        )
        return

    control_cols = st.columns((0.85, 1.25, 0.9))
    severity_options = ["All", "Severe", "Moderate", "Watch", "Background"]
    severity = control_cols[0].selectbox("Severity", severity_options)
    lst_range = control_cols[1].slider(
        "LST range (C)",
        min_value=float(points["LST_C"].min()),
        max_value=float(points["LST_C"].max()),
        value=(float(points["LST_C"].min()), float(points["LST_C"].max())),
    )
    max_points = control_cols[2].slider(
        "Displayed points",
        min_value=100,
        max_value=min(5000, len(points)),
        value=min(1800, len(points)),
        step=100,
    )

    filtered = points[(points["LST_C"] >= lst_range[0]) & (points["LST_C"] <= lst_range[1])]
    if severity != "All":
        filtered = filtered[filtered["hotspot_label"] == severity]
    filtered = filtered.sort_values("hotspot_score", ascending=False).head(max_points)

    top_cols = st.columns((1.25, 0.75))
    with top_cols[0]:
        render_section_header(
            "Spatial Heat Surface",
            "A clean, judge-friendly view of hotspot intensity derived from the sampled artifact.",
        )
        if filtered.empty:
            st.warning("No points matched the current filter combination.")
        else:
            st_folium(_build_hotspot_map(filtered, radius=5.0), use_container_width=True, height=600)
            render_legend(
                [
                    ("Background", HOTSPOT_COLORS["Background"]),
                    ("Watch", HOTSPOT_COLORS["Watch"]),
                    ("Moderate", HOTSPOT_COLORS["Moderate"]),
                    ("Severe", HOTSPOT_COLORS["Severe"]),
                ]
            )
    with top_cols[1]:
        render_section_header(
            "Analyst Summary",
            "Use these cards as a spoken briefing while the map is on screen.",
        )
        render_kpi_card("Displayed Points", f"{len(filtered):,}", "Points shown after the active severity and LST filters.", accent="linear-gradient(90deg, #2563eb, #38bdf8)")
        mean_lst = filtered["LST_C"].mean() if not filtered.empty else float("nan")
        render_kpi_card("Mean Displayed LST", _format_value(mean_lst, suffix=" C"), "Average land-surface temperature of the current view.", accent="linear-gradient(90deg, #f59e0b, #ef4444)")
        hotspot_share = filtered["is_hotspot"].mean() * 100 if not filtered.empty else float("nan")
        render_kpi_card("Hotspot Share", _format_value(hotspot_share, suffix="%"), "Fraction of displayed points above background heat conditions.", accent="linear-gradient(90deg, #0f9f6e, #2563eb)")
        peak_score = filtered["hotspot_score"].max() if not filtered.empty else float("nan")
        render_kpi_card("Peak Hotspot Score", _format_value(peak_score), "Highest relative hotspot intensity in the current view.", accent="linear-gradient(90deg, #f97316, #dc2626)")

    if not filtered.empty:
        lower_cols = st.columns((1, 1))
        with lower_cols[0]:
            render_insight_card(
                "Heat narrative",
                _build_hotspot_page_copy(filtered, severity),
                tone="orange",
            )
        with lower_cols[1]:
            render_dataframe_preview(
                filtered[["latitude", "longitude", "LST_C", "hotspot_label", "hotspot_score"]],
                "Filtered Hotspot Sample",
                limit=12,
                caption="Use this table when a judge asks for concrete point-level evidence.",
            )


def _build_hotspot_map(points: pd.DataFrame, radius: float = 5.0) -> folium.Map:
    center = [points["latitude"].mean(), points["longitude"].mean()]
    hotspot_map = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
    cluster = MarkerCluster().add_to(hotspot_map)

    for row in points.itertuples(index=False):
        color = HOTSPOT_COLORS.get(row.hotspot_label, HOTSPOT_COLORS["Background"])
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=1.2,
            popup=(
                f"LST: {row.LST_C:.2f} C<br>"
                f"Severity: {row.hotspot_label}<br>"
                f"Hotspot score: {row.hotspot_score:.1f}"
            ),
        ).add_to(cluster)

    return hotspot_map


def render_ward_rankings_page(artifacts: DashboardArtifacts) -> None:
    render_hero_banner(
        "Ward Rankings",
        "Executive prioritization for ward-level heat intervention planning.",
        "Decision Intelligence",
        ["Ward choropleth", "Ranking table", "High-risk selection brief"],
    )

    points = load_training_points(artifacts)
    wards = load_ward_boundaries(artifacts)
    precomputed = load_ward_summaries(artifacts)

    if not precomputed.empty and wards.empty:
        render_empty_state(
            "Ward summary available without geometry",
            "A summary artifact exists, but there is no ward boundary GeoJSON for the map layer.",
            ["Add a local ward boundary file to unlock the full map-first ranking experience."],
        )
        st.dataframe(precomputed.sort_values("rank"), use_container_width=True, hide_index=True)
        return

    if wards.empty:
        render_empty_state(
            "Ward rankings pending",
            "Add a local ward boundary GeoJSON artifact to enable choropleths and ward comparisons.",
            [
                "Place a ward file at `data/external/ward_boundaries.geojson`, `data/external/kmc_wards.geojson`, or `data/external/wards.geojson`.",
                "Keep one stable ward identifier and one display name field; the loader will normalize them to `ward_id` and `ward_name`.",
            ],
        )
        return

    ward_summary = derive_ward_summary(points, wards) if precomputed.empty else wards.merge(precomputed, on=["ward_id", "ward_name"], how="left")
    if ward_summary.empty or ward_summary["sample_count"].fillna(0).sum() == 0:
        render_empty_state(
            "Ward polygons loaded, but no mapped points",
            "The geometry is available, yet no points could be assigned to wards from the current artifact set.",
            ["Check whether the point sample overlaps the ward polygons and uses the same coordinate reference system."],
        )
        return

    ranked = (
        ward_summary.drop(columns="geometry")
        .sort_values(["rank", "hotspot_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    selected_ward_name = st.selectbox("Inspect ward", ranked["ward_name"].dropna().tolist())
    selected = ranked[ranked["ward_name"] == selected_ward_name].iloc[0]

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_kpi_card("Ward Rank", str(int(selected["rank"])) if pd.notna(selected["rank"]) else "N/A", "Relative priority among wards based on hotspot score.", accent="linear-gradient(90deg, #2563eb, #38bdf8)")
    with kpi_cols[1]:
        render_kpi_card("Mean LST", _format_value(selected["mean_lst"], suffix=" C"), "Average land surface temperature inside the selected ward.", accent="linear-gradient(90deg, #f59e0b, #ef4444)")
    with kpi_cols[2]:
        render_kpi_card("Hotspot Share", _format_value(selected["hotspot_share"] * 100 if pd.notna(selected["hotspot_share"]) else float('nan'), suffix="%"), "Share of ward sample points above background heat conditions.", accent="linear-gradient(90deg, #0f9f6e, #2563eb)")
    with kpi_cols[3]:
        render_kpi_card("Sample Points", f"{int(selected['sample_count']):,}", "Number of sampled points contributing to the selected ward signal.", accent="linear-gradient(90deg, #7c3aed, #2563eb)")

    layout_cols = st.columns((1.18, 0.82))
    with layout_cols[0]:
        render_section_header(
            "Ward Heat Prioritization Map",
            "The choropleth makes high-risk wards presentation-ready while preserving exact ranking behavior.",
        )
        st_folium(_build_ward_map(ward_summary, selected["ward_id"]), use_container_width=True, height=600)
    with layout_cols[1]:
        render_section_header(
            "Selected Ward Brief",
            "A concise briefing card that makes the ranking easier to speak through in a demo.",
        )
        render_insight_card(
            "Priority recommendation",
            _build_ward_recommendation_copy(selected),
            tone="blue",
        )
        render_insight_card(
            "Ranking logic",
            "Ward ranking is driven by a composite hotspot score that blends mean land-surface temperature with the share of points above background heat conditions.",
            tone="green",
        )

    display = ranked[["rank", "ward_name", "mean_lst", "hotspot_share", "hotspot_score", "sample_count"]].copy()
    display["hotspot_share"] = (display["hotspot_share"] * 100).round(1)
    render_dataframe_preview(
        display,
        "Ward Ranking Table",
        limit=len(display),
        caption="Ranking formula: hotspot_score = 0.6 × mean_lst + 0.4 × hotspot_share × 100.",
    )


def _build_ward_map(ward_summary: gpd.GeoDataFrame, selected_ward_id: str) -> folium.Map:
    centroid = ward_summary.geometry.unary_union.centroid
    ward_map = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles="CartoDB positron")

    high_cutoff = ward_summary["hotspot_score"].quantile(0.75) if "hotspot_score" in ward_summary else None

    def style_function(feature: dict) -> dict:
        ward_id = str(feature["properties"].get("ward_id"))
        is_selected = ward_id == str(selected_ward_id)
        score = feature["properties"].get("hotspot_score")
        fill_color = "#dbeafe"
        if pd.notna(score):
            if high_cutoff is not None and score >= high_cutoff:
                fill_color = "#fb923c"
            else:
                fill_color = "#93c5fd"
        if is_selected:
            fill_color = "#2563eb"
        return {
            "fillColor": fill_color,
            "color": "#0f172a",
            "weight": 2.4 if is_selected else 1.1,
            "fillOpacity": 0.68,
        }

    tooltip = folium.features.GeoJsonTooltip(
        fields=["ward_name", "rank", "mean_lst", "hotspot_share", "sample_count"],
        aliases=["Ward", "Rank", "Mean LST", "Hotspot share", "Sample count"],
        localize=True,
        style="background-color: white; color: #0f172a; border: 1px solid #dbeafe; border-radius: 8px; padding: 8px;",
    )

    folium.GeoJson(ward_summary.to_json(), style_function=style_function, tooltip=tooltip).add_to(ward_map)
    return ward_map


def render_driver_explanations_page(artifacts: DashboardArtifacts) -> None:
    render_hero_banner(
        "Driver Explanations",
        "Narrative-first intelligence for why heat stress is elevated.",
        "Explainable AI Layer",
        ["Global SHAP signals", "Ward-level insight cards", "Judge-friendly explanation flow"],
    )

    points = load_training_points(artifacts)
    shap_df = load_shap_values(artifacts)
    wards = load_ward_boundaries(artifacts)

    if shap_df.empty:
        render_empty_state(
            "Driver explanations unavailable",
            "No SHAP artifact was found yet, so the insight surface cannot render.",
            [
                "Train the baseline model and run `python -m src.explainability.shap_baseline`.",
                "Ensure `outputs/shap_values.csv` exists, then refresh the dashboard.",
            ],
        )
        return

    summary = summarize_global_shap(shap_df)
    merged = merge_shap_with_points(points, shap_df)
    ward_summary = derive_ward_summary(points, wards, shap_df) if not wards.empty and not merged.empty else gpd.GeoDataFrame()

    top_feature_label = summary.iloc[0]["feature_label"] if not summary.empty else "Driver signal"
    top_feature_value = summary.iloc[0]["mean_abs_shap"] if not summary.empty else float("nan")

    cards = st.columns(3)
    with cards[0]:
        render_kpi_card("Leading Global Driver", top_feature_label, "Most influential driver in the current SHAP artifact.", accent="linear-gradient(90deg, #2563eb, #7c3aed)")
    with cards[1]:
        render_kpi_card("Mean SHAP Magnitude", _format_value(top_feature_value), "Average absolute explanatory contribution of the leading driver.", accent="linear-gradient(90deg, #0f9f6e, #2563eb)")
    with cards[2]:
        ward_ready = not ward_summary.empty and ward_summary["dominant_driver"].notna().any()
        render_kpi_card("Ward Insight Coverage", "Ready" if ward_ready else "Global only", "Ward-level explanation cards activate when SHAP aligns with ward-mapped points.", accent="linear-gradient(90deg, #f97316, #dc2626)")

    top_cols = st.columns((0.95, 1.05))
    with top_cols[0]:
        render_section_header(
            "Global Driver Ranking",
            "The chart supports the narrative; the explanation cards carry the main message.",
        )
        st.bar_chart(summary.set_index("feature_label")["mean_abs_shap"].head(12))
    with top_cols[1]:
        render_section_header(
            "Insight Cards",
            "These turn raw SHAP values into explanations that sound like insights, not debug output.",
        )
        if ward_ready:
            options = ward_summary["ward_name"].dropna().tolist()
            ward_name = st.selectbox("Explain ward", options)
            ward_row = ward_summary[ward_summary["ward_name"] == ward_name].iloc[0]
            dominant_label = get_driver_label(ward_row.get("dominant_driver"))
            render_insight_card(
                "Ward explanation",
                (
                    f"High built-up density and low vegetation cover are the dominant contributors to elevated heat stress in {ward_name}."
                    if dominant_label in {"Built-up intensity", "Built fraction", "Non-residential built fraction", "Vegetation cover"}
                    else f"{dominant_label} is the strongest explanatory signal for elevated heat stress in {ward_name}."
                ),
                tone="blue",
            )
            render_insight_card(
                "Context",
                (
                    f"{ward_name} ranks #{int(ward_row['rank'])} with a mean LST of {ward_row['mean_lst']:.2f} C "
                    f"and a hotspot share of {ward_row['hotspot_share'] * 100:.1f}%."
                ),
                tone="orange",
            )
        else:
            render_insight_card(
                "Global explanation",
                f"The strongest current driver signal across the loaded artifact set is {top_feature_label}.",
                tone="blue",
            )
            render_insight_card(
                "Why this matters",
                "This is the explanatory layer judges can use to move from where the hotspots are to why those hotspots are forming.",
                tone="green",
            )

    with st.expander("See ranked SHAP driver table", expanded=False):
        st.dataframe(summary.head(15), use_container_width=True, hide_index=True)


def render_scenarios_page(artifacts: DashboardArtifacts) -> None:
    render_hero_banner(
        "Scenario Simulation",
        "Transparent intervention impact estimates using model outputs, SHAP signals, and bounded domain assumptions.",
        "Intervention Simulator",
        ["Ward controls", "Cooling estimate", "Confidence notes", "PINN-ready response model"],
    )

    planning_wards = _load_planning_wards(artifacts)
    shap_summary = summarize_global_shap(load_shap_values(artifacts))
    if planning_wards.empty:
        render_empty_state(
            "Scenario simulation unavailable",
            "Ward heat summaries are needed before intervention impacts can be estimated.",
            [
                "Run `python -m src.feature_engineering.ward_aggregation` to produce `outputs/ward_heat_summary.geojson`.",
                "Reload the dashboard after the ward summary artifact is available.",
            ],
        )
        return

    ranked_wards = planning_wards.sort_values("hotspot_score", ascending=False).reset_index(drop=True)
    control_cols = st.columns((1.0, 1.2, 0.85, 0.85))
    intervention = control_cols[0].selectbox(
        "Intervention type",
        list(SUPPORTED_INTERVENTIONS),
        key="scenario_intervention_type",
    )
    selected_ward_name = control_cols[1].selectbox("Target ward", ranked_wards["ward_name"].dropna().tolist())
    intensity = control_cols[2].slider("Intensity", min_value=10, max_value=100, value=65, step=5)
    coverage = control_cols[3].slider("Coverage", min_value=5, max_value=80, value=25, step=5)

    selected = ranked_wards[ranked_wards["ward_name"] == selected_ward_name].iloc[0]
    result = simulate_intervention(selected.to_dict(), intervention, intensity, coverage, shap_summary)

    body_cols = st.columns((0.95, 1.05))
    with body_cols[0]:
        render_section_header(
            "Simulation Output",
            "Cooling impact is directional and bounded; confidence reflects artifact support and intervention suitability.",
        )
        kpis = st.columns(2)
        with kpis[0]:
            render_kpi_card("Estimated Cooling", _format_value(result.estimated_cooling_c, suffix=" C"), "Projected ward-scale LST reduction.", accent="linear-gradient(90deg, #0f9f6e, #2563eb)")
        with kpis[1]:
            render_kpi_card("Affected Area", _format_value(result.affected_area_km2, suffix=" km2"), "Estimated area touched by selected coverage.", accent="linear-gradient(90deg, #2563eb, #38bdf8)")
        kpis = st.columns(2)
        with kpis[0]:
            render_kpi_card("Confidence", result.confidence, f"Score {result.confidence_score:.2f} from data support and suitability.", accent="linear-gradient(90deg, #f97316, #dc2626)")
        with kpis[1]:
            render_kpi_card("Coverage", f"{result.coverage_pct:.0f}%", "Share of ward area selected for intervention.", accent="linear-gradient(90deg, #7c3aed, #2563eb)")
    with body_cols[1]:
        render_section_header(
            "Transparent Notes",
            "These notes explain what the simulator used and where uncertainty remains.",
        )
        render_insight_card("Implementation notes", " ".join(result.implementation_notes), tone="blue")
        render_insight_card("Domain assumptions", " ".join(result.assumptions), tone="green")
        st.json(result.to_dict(), expanded=False)

    comparison = simulate_for_wards(ranked_wards.head(12), intervention, intensity, coverage, shap_summary)
    if not comparison.empty:
        display = comparison[["ward_name", "estimated_cooling_c", "affected_area_km2", "confidence", "confidence_score"]].sort_values(
            "estimated_cooling_c",
            ascending=False,
        )
        render_dataframe_preview(
            display,
            "Compare Top Wards For This Intervention",
            limit=12,
            caption="Same intervention settings applied to the current highest-risk wards.",
        )

def render_optimization_page(artifacts: DashboardArtifacts) -> None:
    render_hero_banner(
        "Budget Optimization",
        "Greedy baseline allocation that maximizes estimated cooling under a planner-defined budget.",
        "Constrained Planning",
        ["Budget controls", "Ward allocation", "Cost breakdown", "Explainable ranking"],
    )

    planning_wards = _load_planning_wards(artifacts)
    shap_summary = summarize_global_shap(load_shap_values(artifacts))
    if planning_wards.empty:
        render_empty_state(
            "Optimization unavailable",
            "The optimizer needs ward heat summaries before it can rank budget allocations.",
            ["Generate `outputs/ward_heat_summary.geojson` with the ward aggregation pipeline."],
        )
        return

    ranked_wards = planning_wards.sort_values("hotspot_score", ascending=False).reset_index(drop=True)
    ward_options = ranked_wards["ward_name"].dropna().tolist()
    control_cols = st.columns((0.75, 1.25, 0.7, 0.7))
    budget_crore = control_cols[0].number_input("Total budget (INR crore)", min_value=0.5, max_value=200.0, value=25.0, step=0.5)
    targets = control_cols[1].multiselect("Target wards", ward_options, default=ward_options[: min(8, len(ward_options))])
    intensity = control_cols[2].slider("Intensity", min_value=20, max_value=100, value=70, step=5, key="opt_intensity")
    coverage = control_cols[3].slider("Coverage", min_value=5, max_value=60, value=20, step=5, key="opt_coverage")

    with st.expander("Intervention cost assumptions", expanded=False):
        cost_inputs: dict[str, float] = {}
        cost_cols = st.columns(2)
        for index, intervention in enumerate(SUPPORTED_INTERVENTIONS):
            default_crore = DEFAULT_INTERVENTION_COSTS_INR_PER_KM2[intervention] / 10_000_000.0
            value = cost_cols[index % 2].number_input(
                f"{intervention} cost (INR crore/km2)",
                min_value=0.1,
                max_value=200.0,
                value=float(default_crore),
                step=0.5,
                key=f"cost_{intervention}",
            )
            cost_inputs[intervention] = value * 10_000_000.0

    optimizer = GreedyCoolingOptimizer(intensity=float(intensity), coverage_pct=float(coverage))
    result = optimize_cooling_plan(
        ranked_wards,
        total_budget=float(budget_crore) * 10_000_000.0,
        intervention_costs=cost_inputs,
        target_wards=targets,
        shap_summary=shap_summary,
        strategy=optimizer,
    )
    st.session_state["uc_last_optimization_plan"] = result.ranked_plan

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_kpi_card("Allocated Budget", _format_inr_crore(result.allocated_budget_inr), "Selected interventions within the budget.", accent="linear-gradient(90deg, #2563eb, #38bdf8)")
    with kpi_cols[1]:
        render_kpi_card("Remaining Budget", _format_inr_crore(result.remaining_budget_inr), "Unallocated budget after greedy selection.", accent="linear-gradient(90deg, #0f9f6e, #2563eb)")
    with kpi_cols[2]:
        render_kpi_card("Plan Items", str(len(result.ranked_plan)), "One selected intervention per ward at most.", accent="linear-gradient(90deg, #f97316, #dc2626)")
    with kpi_cols[3]:
        render_kpi_card("Total Cooling", _format_value(result.estimated_total_cooling_c, suffix=" C"), "Sum of ward-scale cooling estimates.", accent="linear-gradient(90deg, #7c3aed, #2563eb)")

    render_insight_card("Optimization logic", result.explanation, tone="blue")
    if result.ranked_plan.empty:
        st.warning("No candidate fit inside the selected budget. Increase the budget or reduce coverage.")
    else:
        plan_display = result.ranked_plan[
            ["rank", "ward_name", "intervention_type", "estimated_cooling_c", "affected_area_km2", "estimated_cost_inr", "confidence", "rationale"]
        ].copy()
        plan_display["estimated_cost_inr"] = plan_display["estimated_cost_inr"].map(_format_inr_crore)
        render_dataframe_preview(plan_display, "Ranked Intervention Plan", limit=len(plan_display), caption="Greedy baseline allocation sorted by cooling efficiency.")
        st.bar_chart(result.ranked_plan.set_index("ward_name")["estimated_cooling_c"])

    with st.expander("Candidate ranking table", expanded=False):
        candidates = result.candidate_table[
            ["ward_name", "intervention_type", "estimated_cooling_c", "estimated_cost_inr", "cooling_efficiency", "confidence"]
        ].copy()
        candidates["estimated_cost_inr"] = candidates["estimated_cost_inr"].map(_format_inr_crore)
        st.dataframe(candidates.head(40), use_container_width=True, hide_index=True)


def render_ai_recommendations_page(artifacts: DashboardArtifacts) -> None:
    render_hero_banner(
        "AI Recommendations",
        "Deterministic planner-friendly summaries grounded in ward metrics, SHAP drivers, and optimization outputs.",
        "Decision Support",
        ["Ward summaries", "Recommendations", "City briefing", "Offline templates"],
    )

    planning_wards = _load_planning_wards(artifacts)
    shap_df = load_shap_values(artifacts)
    shap_summary = summarize_global_shap(shap_df)
    optimization_plan = st.session_state.get("uc_last_optimization_plan", pd.DataFrame())
    if planning_wards.empty:
        render_empty_state(
            "Recommendations unavailable",
            "The decision-support layer needs ward-level heat summaries before it can generate grounded narratives.",
            ["Generate `outputs/ward_heat_summary.geojson`, then return to this page."],
        )
        return

    ranked_wards = planning_wards.sort_values("hotspot_score", ascending=False).reset_index(drop=True)
    city_brief = generate_city_briefing(ranked_wards, shap_df, optimization_plan)
    render_insight_card("City-level briefing", city_brief, tone="blue")

    control_cols = st.columns((1.1, 0.5))
    selected_ward_name = control_cols[0].selectbox("Ward summary", ranked_wards["ward_name"].dropna().tolist(), key="ai_ward")
    limit = control_cols[1].slider("Recommendation count", min_value=3, max_value=15, value=8, step=1)
    selected = ranked_wards[ranked_wards["ward_name"] == selected_ward_name].iloc[0]
    ward_brief = generate_ward_summary(selected, shap_summary, optimization_plan)

    cols = st.columns((0.9, 1.1))
    with cols[0]:
        render_section_header("Selected Ward Brief", "Generated from deterministic templates; no external LLM API is used.")
        render_insight_card("Ward summary", ward_brief.summary, tone="orange")
        render_insight_card("Recommendation", ward_brief.recommendation, tone="green")
    with cols[1]:
        render_dataframe_preview(
            pd.DataFrame({"Evidence": list(ward_brief.evidence)}),
            "Grounding Evidence",
            limit=10,
            caption="Metrics used by the planner summary.",
        )

    recommendations = generate_intervention_recommendations(ranked_wards, shap_summary, optimization_plan, limit=limit)
    render_dataframe_preview(
        recommendations,
        "Planner Recommendation Table",
        limit=len(recommendations),
        caption="Reusable narrative snippets for the highest-priority wards.",
    )


def _render_readiness_strip(status: dict) -> None:
    cols = st.columns(3)
    with cols[0]:
        render_status_pill(
            "Heat hotspot visualization",
            status["training_points_ready"],
            "Ready from the sampled training artifact." if status["training_points_ready"] else "Waiting for sampled point artifact.",
        )
    with cols[1]:
        render_status_pill(
            "Ward rankings",
            status["ward_boundaries_ready"],
            "Auto-enables once local ward geometry is available." if not status["ward_boundaries_ready"] else "Ward boundaries are loaded for prioritization.",
        )
    with cols[2]:
        render_status_pill(
            "Driver explanations",
            status["shap_ready"],
            "Global and ward insight cards activate when SHAP outputs are available." if not status["shap_ready"] else "Explanation layer is ready for narrative insights.",
        )


def _resolve_overview_driver(artifacts: DashboardArtifacts, points: pd.DataFrame, wards: gpd.GeoDataFrame) -> str:
    shap_df = load_shap_values(artifacts)
    if shap_df.empty:
        return "Awaiting SHAP outputs"

    summary = summarize_global_shap(shap_df)
    if summary.empty:
        return "Driver signal unavailable"

    merged = merge_shap_with_points(points, shap_df)
    if not wards.empty and not merged.empty:
        ward_summary = derive_ward_summary(points, wards, shap_df)
        if not ward_summary.empty and ward_summary["dominant_driver"].notna().any():
            return get_driver_label(ward_summary.dropna(subset=["dominant_driver"]).iloc[0]["dominant_driver"])

    return str(summary.iloc[0]["feature_label"])


def _format_coverage(status: dict) -> str:
    score = sum(
        [
            1 if status["training_points_ready"] else 0,
            1 if status["shap_ready"] else 0,
            1 if status["ward_boundaries_ready"] else 0,
        ]
    )
    return f"{score}/3"


def _build_hotspot_overview_copy(points: pd.DataFrame) -> str:
    severe = int((points["hotspot_label"] == "Severe").sum())
    moderate = int((points["hotspot_label"] == "Moderate").sum())
    return (
        f"The current sample contains {severe:,} severe hotspots and {moderate:,} moderate hotspots, "
        "giving judges a fast view of where elevated heat stress is clustering."
    )


def _build_artifact_health_copy(status: dict) -> str:
    ready = []
    if status["training_points_ready"]:
        ready.append("sampled heat points")
    if status["shap_ready"]:
        ready.append("SHAP explanations")
    if status["ward_boundaries_ready"]:
        ready.append("ward geometry")
    if not ready:
        return "No supporting artifacts are loaded yet."
    return "Loaded artifacts: " + ", ".join(ready) + "."


def _build_hotspot_page_copy(filtered: pd.DataFrame, severity: str) -> str:
    scope = "the full point set" if severity == "All" else f"the {severity.lower()} class"
    peak = filtered.sort_values("hotspot_score", ascending=False).iloc[0]
    return (
        f"This view focuses on {scope}. The strongest visible hotspot is currently near "
        f"({peak['latitude']:.3f}, {peak['longitude']:.3f}) with an LST of {peak['LST_C']:.2f} C."
    )


def _build_ward_recommendation_copy(selected: pd.Series) -> str:
    ward_name = selected["ward_name"]
    rank = int(selected["rank"]) if pd.notna(selected["rank"]) else None
    mean_lst = selected["mean_lst"]
    hotspot_share = selected["hotspot_share"] * 100 if pd.notna(selected["hotspot_share"]) else float("nan")
    return (
        f"{ward_name} is a top-{rank} candidate for intervention review, with a mean LST of "
        f"{mean_lst:.2f} C and {hotspot_share:.1f}% of sampled points above background heat conditions."
    )


def _load_planning_wards(artifacts: DashboardArtifacts) -> pd.DataFrame:
    precomputed = load_ward_summaries(artifacts)
    if not precomputed.empty:
        return _prepare_planning_wards(precomputed)

    points = load_training_points(artifacts)
    wards = load_ward_boundaries(artifacts)
    shap_df = load_shap_values(artifacts)
    if points.empty or wards.empty:
        return pd.DataFrame()
    derived = derive_ward_summary(points, wards, shap_df)
    if derived.empty:
        return pd.DataFrame()
    return _prepare_planning_wards(pd.DataFrame(derived.drop(columns="geometry", errors="ignore")))


def _prepare_planning_wards(wards: pd.DataFrame) -> pd.DataFrame:
    planning = wards.copy()
    if "mean_lst" not in planning and "PREDICTED_LST_C" in planning:
        planning["mean_lst"] = planning["PREDICTED_LST_C"]
    if "PREDICTED_LST_C" not in planning and "mean_lst" in planning:
        planning["PREDICTED_LST_C"] = planning["mean_lst"]
    if "ward_name" not in planning and "WARD" in planning:
        planning["ward_name"] = "Ward " + planning["WARD"].astype(str)
    if "ward_id" not in planning and "WARD" in planning:
        planning["ward_id"] = planning["WARD"].astype(str)
    if "hotspot_score" not in planning and "mean_lst" in planning:
        planning["hotspot_score"] = planning["mean_lst"].rank(pct=True) * 100.0
    if "hotspot_share" not in planning:
        planning["hotspot_share"] = planning["hotspot_score"].rank(pct=True) if "hotspot_score" in planning else 0.0
    if "rank" not in planning and "hotspot_score" in planning:
        planning["rank"] = planning["hotspot_score"].rank(ascending=False, method="dense")
    if "sample_count" not in planning and "pixel_count" in planning:
        planning["sample_count"] = planning["pixel_count"]
    planning = planning.dropna(subset=["ward_name"])
    if "hotspot_score" in planning:
        planning = planning.sort_values("hotspot_score", ascending=False)
    return planning.reset_index(drop=True)


def _format_inr_crore(value: float) -> str:
    return f"INR {float(value) / 10_000_000.0:.2f} cr"


def _format_value(value: float | str, suffix: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "N/A"
    try:
        if isnan(float(value)):
            return "N/A"
    except (TypeError, ValueError):
        return "N/A"
    return f"{float(value):.1f}{suffix}" if suffix == "%" else f"{float(value):.2f}{suffix}"
