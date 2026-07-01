# UrbanCool AI Dashboard Walkthrough

This guide explains how to use the UrbanCool AI dashboard as a first-time user. It is written for hackathon judges, reviewers, planners, and collaborators who want to understand the product flow without reading the source code first.

## Dashboard Overview

The dashboard turns model artifacts into an interactive urban heat decision-support workflow. It helps answer four practical questions:

- Where are the heat hotspots?
- Which wards should be prioritized?
- What is driving elevated heat stress?
- Which cooling interventions should be selected under budget constraints?

Start the dashboard with:

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

Navigation is handled from the left sidebar under **Platform Modules**. Each page is designed to work independently, but the recommended review path is:

```text
Overview -> Hotspots Map -> Ward Rankings -> Driver Explanations -> Scenario Simulation -> Optimization -> AI Recommendations
```

If an artifact is missing, the dashboard shows a readiness message instead of inventing outputs.

![Screenshot placeholder: dashboard overview](screenshots/dashboard_overview.png)

## Page-by-Page Explanation

### 1. Overview

The **Overview** page is the executive entry point. It summarizes the platform status, shows a hotspot preview map, and presents the main signals needed for a short demo.

Use this page to:

- confirm whether training points, SHAP outputs, and ward geometry are loaded
- see high-level hotspot counts
- understand the current dominant heat driver
- introduce the full workflow to a reviewer

Recommended narration:

> UrbanCool AI starts with geospatial heat evidence, explains the drivers behind hotspots, then turns those insights into ward-level intervention planning.

![Screenshot placeholder: overview page](screenshots/overview_page.png)

### 2. Hotspots Map

The **Hotspots Map** page shows sampled land-surface temperature points on an interactive map. Points are classified into severity categories so users can quickly inspect where high heat stress clusters.

Controls:

- **Severity:** filter by `All`, `Severe`, `Moderate`, `Watch`, or `Background`
- **LST range:** restrict the map to a temperature interval
- **Displayed points:** limit the number of rendered points for performance

Outputs:

- interactive map of heat samples
- displayed point count
- mean displayed LST
- hotspot share
- peak hotspot score
- filtered evidence table

![Screenshot placeholder: hotspots map](screenshots/hotspots_map.png)

### 3. Ward Rankings

The **Ward Rankings** page translates point-level heat evidence into ward-level prioritization. This is the main planning view for deciding which wards deserve attention first.

Controls:

- **Inspect ward:** select a ward from the ranking list

Outputs:

- selected ward rank
- mean LST
- hotspot share
- sample count
- ward choropleth map
- selected ward brief
- full ranking table

Use this page when a reviewer asks:

- Which ward is hottest?
- How are wards prioritized?
- Can I compare wards directly?

![Screenshot placeholder: ward rankings](screenshots/ward_rankings.png)

### 4. Driver Explanations

The **Driver Explanations** page uses SHAP outputs to show why the model predicts elevated heat. It connects hotspots to interpretable urban features.

Outputs:

- leading global SHAP driver
- SHAP magnitude summary
- ward insight coverage status
- global driver bar chart
- ward-level explanation cards when ward-mapped SHAP evidence is available
- ranked SHAP driver table

Common driver interpretations:

- high `NDBI` or `BUILT_FRACTION`: built-up intensity is contributing to heat
- low `NDVI`: vegetation cooling is limited
- low or changing `DW_WATER_PROB`: blue-space cooling may be limited
- high morphology features: built form may be affecting heat retention

![Screenshot placeholder: driver explanations](screenshots/driver_explanations.png)

### 5. Scenario Simulation

The **Scenario Simulation** page estimates the likely directional cooling impact of one intervention in one ward. The simulator uses existing model outputs, SHAP support, feature sensitivity, and documented domain assumptions.

Controls:

- **Intervention type:** choose from Urban Greening, Cool Roofs, Reflective / High-Albedo Surfaces, Blue-Green Infrastructure, or Water Body Restoration
- **Target ward:** choose the ward to simulate
- **Intensity:** adjust intervention strength
- **Coverage:** adjust the share of ward area affected

Outputs:

- estimated cooling impact
- affected area
- confidence level
- confidence score
- implementation notes
- domain assumptions
- JSON output contract
- comparison table across top wards for the selected intervention

Important interpretation note:

The scenario engine provides bounded, ward-scale planning estimates. It does not claim exact microclimate cooling at street or building scale.

![Screenshot placeholder: scenario simulation](screenshots/scenario_simulation.png)

### 6. Optimization

The **Optimization** page builds a budget-constrained intervention plan. It uses the scenario engine to generate ward-intervention candidates, then ranks them by estimated cooling per rupee.

Controls:

- **Total budget:** set available funding in INR crore
- **Target wards:** choose wards eligible for allocation
- **Intensity:** set intervention intensity used for candidate generation
- **Coverage:** set intervention coverage used for candidate generation
- **Intervention cost assumptions:** edit estimated INR crore per square kilometer for each intervention type

Outputs:

- allocated budget
- remaining budget
- number of plan items
- estimated total cooling
- optimization logic card
- ranked intervention plan
- cooling impact chart
- candidate ranking table

The current optimizer is a greedy baseline. It is intentionally transparent and easy to explain during a hackathon review.

![Screenshot placeholder: optimization page](screenshots/optimization_page.png)

### 7. AI Recommendations

The **AI Recommendations** page converts model, SHAP, scenario, and optimization outputs into planner-friendly language. It uses deterministic templates and does not require an external LLM API.

Controls:

- **Ward summary:** choose a ward to explain
- **Recommendation count:** select how many ward recommendations to show

Outputs:

- city-level briefing
- selected ward summary
- selected ward recommendation
- grounding evidence table
- planner recommendation table

Example output:

> Ward 37 exhibits elevated heat stress driven primarily by built-up density and low vegetation cover. Urban greening is expected to provide the strongest near-term marginal benefit for this ward.

![Screenshot placeholder: AI recommendations](screenshots/ai_recommendations.png)

## Hotspot Workflow

Use this workflow when the goal is to identify spatial heat clusters.

1. Open **Overview** and confirm the sampled training artifact is ready.
2. Go to **Hotspots Map**.
3. Set **Severity** to `Severe`.
4. Adjust **Displayed points** if the map feels crowded.
5. Inspect point clusters and hover/click points for LST and hotspot score.
6. Use the filtered table to cite concrete point-level evidence.
7. Move to **Ward Rankings** to convert hotspots into administrative priorities.

Screenshot placeholders:

![Screenshot placeholder: severe hotspot filter](screenshots/hotspot_severe_filter.png)
![Screenshot placeholder: hotspot evidence table](screenshots/hotspot_evidence_table.png)

## Ward Ranking Workflow

Use this workflow when the goal is to decide which wards should be prioritized.

1. Open **Ward Rankings**.
2. Select a high-ranked ward from **Inspect ward**.
3. Review rank, mean LST, hotspot share, and sample count.
4. Use the choropleth to compare the selected ward against neighboring wards.
5. Read the selected ward brief.
6. Review the ranking table for top candidates.
7. Continue to **Driver Explanations** to understand why the ward is hot.

Screenshot placeholders:

![Screenshot placeholder: selected ward choropleth](screenshots/selected_ward_choropleth.png)
![Screenshot placeholder: ward ranking table](screenshots/ward_ranking_table.png)

## Driver Explanation Workflow

Use this workflow when the goal is to explain model behavior.

1. Open **Driver Explanations**.
2. Start with the leading global driver KPI.
3. Inspect the global SHAP bar chart.
4. If ward-level coverage is ready, select a ward to explain.
5. Read the ward explanation and context cards.
6. Open the SHAP driver table for detailed feature ranking.
7. Use the dominant drivers to choose a relevant scenario intervention.

Screenshot placeholders:

![Screenshot placeholder: global SHAP chart](screenshots/global_shap_chart.png)
![Screenshot placeholder: ward explanation cards](screenshots/ward_explanation_cards.png)

## Scenario Simulation Workflow

Use this workflow when the goal is to test one intervention in one ward.

1. Open **Scenario Simulation**.
2. Choose an intervention type.
3. Choose a target ward.
4. Set intervention intensity.
5. Set intervention coverage.
6. Review estimated cooling, affected area, and confidence.
7. Read the implementation notes and domain assumptions.
8. Compare the same intervention across the top wards using the comparison table.
9. Move to **Optimization** when you want to allocate a budget across multiple wards.

Screenshot placeholders:

![Screenshot placeholder: scenario controls](screenshots/scenario_controls.png)
![Screenshot placeholder: scenario transparent notes](screenshots/scenario_transparent_notes.png)

## Optimization Workflow

Use this workflow when the goal is to produce a budgeted intervention plan.

1. Open **Optimization**.
2. Enter the total budget in INR crore.
3. Select target wards.
4. Set intervention intensity and coverage.
5. Expand **Intervention cost assumptions** if costs need adjustment.
6. Review allocated budget, remaining budget, plan items, and total cooling.
7. Read the optimization logic card.
8. Inspect the ranked intervention plan.
9. Open the candidate ranking table for alternatives that were not selected.
10. Continue to **AI Recommendations** to convert the plan into narrative form.

Screenshot placeholders:

![Screenshot placeholder: optimization controls](screenshots/optimization_controls.png)
![Screenshot placeholder: ranked intervention plan](screenshots/ranked_intervention_plan.png)

## AI Recommendation Workflow

Use this workflow when the goal is to generate reviewer-friendly planning language.

1. Open **AI Recommendations**.
2. Read the city-level briefing first.
3. Select a ward from **Ward summary**.
4. Review the ward summary and recommendation cards.
5. Check the grounding evidence table.
6. Set the recommendation count.
7. Use the planner recommendation table in presentations or reports.

The text is deterministic. Running the page repeatedly with the same artifacts will produce the same recommendations.

Screenshot placeholders:

![Screenshot placeholder: city briefing](screenshots/city_briefing.png)
![Screenshot placeholder: planner recommendation table](screenshots/planner_recommendation_table.png)

## Navigation Guidance for First-Time Users

For a five-minute review:

1. **Overview:** introduce the platform and readiness.
2. **Hotspots Map:** show where heat clusters.
3. **Ward Rankings:** identify priority wards.
4. **Driver Explanations:** explain why those wards are hot.
5. **Scenario Simulation:** test one intervention.
6. **Optimization:** allocate a budget.
7. **AI Recommendations:** summarize the action plan.

For a technical review:

1. Start with **Driver Explanations** to verify SHAP grounding.
2. Move to **Scenario Simulation** and inspect the JSON output contract.
3. Open **Optimization** and compare selected actions against the candidate table.
4. End with **AI Recommendations** to verify that language remains grounded in metrics.

For a planning review:

1. Start with **Ward Rankings**.
2. Use **Scenario Simulation** for ward-level what-if analysis.
3. Use **Optimization** for budget tradeoffs.
4. Use **AI Recommendations** for communication-ready summaries.
