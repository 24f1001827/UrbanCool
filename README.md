# UrbanCool AI

UrbanCool AI is a hackathon prototype for urban heat-stress mapping, explainable hotspot diagnosis, and cooling intervention planning. It combines satellite-derived geospatial features, machine learning, SHAP explanations, ward-level aggregation, scenario simulation, budget optimization, and a Streamlit dashboard into one decision-support workflow for urban climate planning.

The current prototype focuses on Kolkata and produces ward-level intelligence for identifying where heat stress is concentrated, why it is elevated, and which cooling interventions should be prioritized under budget constraints.

## 1. Project Overview

UrbanCool AI helps planners move from heat maps to action plans. The system:

- builds a multi-source geospatial feature stack for urban heat analysis
- trains a baseline land surface temperature model
- explains model behavior using SHAP driver attribution
- aggregates predicted heat stress to municipal wards
- simulates realistic cooling interventions with transparent assumptions
- optimizes intervention allocation under budget constraints
- generates deterministic, planner-friendly recommendation summaries
- presents the full workflow through an interactive dashboard

The project is designed for hackathon review: it is artifact-first, reproducible from local outputs, and honest about assumptions and limitations.

## 2. Problem Statement

Urban heat is not evenly distributed. Dense built-up areas, low vegetation cover, limited water bodies, and urban morphology can create localized heat stress that is difficult to see from city-wide averages.

Decision makers need more than a heat map. They need to know:

- which wards are hottest
- what urban features are driving the heat signal
- where interventions are likely to help
- how to allocate limited budgets
- how to communicate recommendations clearly to non-technical stakeholders

UrbanCool AI addresses this by connecting geospatial modeling, explainability, scenario planning, and budget-aware prioritization in one workflow.

## 3. Solution Architecture

```text
Satellite, reanalysis, and urban morphology data
        |
        v
Google Earth Engine feature stack
        |
        v
Training sample + full-city raster features
        |
        v
Baseline XGBoost LST model
        |
        +--> SHAP explainability
        |
        +--> Full-resolution predicted LST raster
        |
        +--> Ward-level heat summary
                    |
                    v
        Scenario simulation engine
                    |
                    v
        Budget-constrained optimization engine
                    |
                    v
        Deterministic AI decision-support layer
                    |
                    v
        Streamlit dashboard
```

The architecture separates prediction, explanation, simulation, optimization, and communication. This keeps the prototype modular and allows future replacement of the intervention response model with a physics-informed neural network or another calibrated simulator.

## 4. System Components

| Component | Purpose | Key Files |
|---|---|---|
| Data pipeline | Fetches and prepares remote-sensing, land-cover, urban morphology, and optional station data | `src/data_pipeline/` |
| Feature engineering | Builds full raster prediction artifacts and ward-level aggregation | `src/feature_engineering/` |
| Baseline model | Trains an XGBoost land surface temperature regressor | `src/models/baseline_xgboost.py` |
| Experimental PINN | Tests a physics-informed neural network with surface-energy-balance residuals | `src/models/pinn.py` |
| Explainability | Generates SHAP feature attributions and visual artifacts | `src/explainability/shap_baseline.py` |
| Scenario engine | Estimates bounded, transparent intervention cooling impacts | `src/scenario_engine/` |
| Optimization engine | Allocates interventions under budget constraints using a greedy baseline | `src/optimization/` |
| Decision-support layer | Produces deterministic planner summaries and recommendations | `src/llm_layer/` |
| Dashboard | Provides the interactive review and planning interface | `src/dashboard/`, `app.py` |

## 5. Data Sources

UrbanCool AI uses a multi-source feature stack:

- **Landsat 8 Level-2:** land surface temperature, NDVI, NDBI, and albedo proxy
- **Sentinel-2 surface reflectance:** additional spectral indices such as NDVI, NDBI, BSI, and MNDWI
- **Dynamic World:** land-cover probabilities and classes, including water, built, grass, trees, crops, and bare ground
- **GHSL:** building height, built-up fraction, non-residential built fraction, and morphology proxies
- **ERA5-Land:** meteorological and radiation context, including air temperature, dewpoint, wind speed, precipitation, pressure, soil moisture, and heat-flux variables
- **OpenStreetMap:** building and road layers fetched for future morphology enhancement
- **CPCB station CSVs:** optional ingestion script exists for externally downloaded observations
- **Ward boundaries:** local ward GeoJSON used for ward-level aggregation and dashboard mapping

The active prototype relies primarily on remote-sensing and reanalysis features. CPCB and OSM fusion are included as future enhancement paths.

## 6. Modeling Approach

The primary predictive model is an XGBoost regressor trained to estimate land surface temperature (`LST_C`) from geospatial, land-cover, morphology, and radiation features.

Current baseline results on a 100,000-row training sample:

| Split | R2 | RMSE |
|---|---:|---:|
| Train | 0.891 | approximately 0.8-1.0 C |
| Validation | 0.775 | approximately 0.8-1.0 C |
| Test | 0.745 | approximately 0.8-1.0 C |

The model uses a spatial train/validation/test split based on a latitude-longitude grid. Splitting by spatial cells reduces leakage from neighboring pixels and gives a more realistic estimate of generalization.

The project also includes an experimental physics-informed neural network in `src/models/pinn.py`. The PINN uses a surface-energy-balance residual term, but currently underperforms the XGBoost baseline on raw predictive fit. It is retained as a future candidate for scenario extrapolation, where physical consistency may matter more than pure predictive accuracy.

## 7. Explainability Layer

UrbanCool AI uses SHAP values to explain which features drive the model's heat predictions.

Top global SHAP drivers observed in the current artifact set:

1. `BUILT_NRES_FRACTION`
2. `NDBI`
3. `DW_WATER_PROB`
4. `MEAN_HEIGHT_150M`
5. `NET_SOLAR_RADIATION_W_M2`
6. `NDVI`

These drivers are directionally consistent with physical expectations:

- built-up density and non-residential built surfaces are associated with higher heat stress
- vegetation and water signals are associated with cooling
- morphology and radiation features provide additional urban context

The dashboard exposes both global SHAP rankings and ward-level explanation cards. The explanation layer is used by the scenario engine and AI recommendation layer to keep intervention recommendations grounded in model evidence.

## 8. Scenario Simulation Engine

The scenario engine estimates directional cooling impact for candidate interventions. It is transparent by design and does not claim physically impossible precision.

Supported interventions:

- Urban Greening
- Cool Roofs
- Reflective / High-Albedo Surfaces
- Blue-Green Infrastructure
- Water Body Restoration

Inputs:

- ward
- intervention type
- intervention intensity
- intervention coverage percentage

Outputs:

- estimated cooling impact
- affected area
- confidence level
- implementation notes
- drivers used
- documented assumptions

The default response model uses:

- ward-level heat predictions
- NDVI, NDBI, built fraction, water probability, and related features
- SHAP feature importance support
- conservative ward-scale cooling bounds
- documented intervention assumptions

The engine exposes an `InterventionResponseModel` protocol so a future PINN or calibrated physical simulator can replace or augment the current response model.

## 9. Optimization Engine

The optimization engine creates a budget-constrained intervention plan.

Inputs:

- total budget
- intervention costs
- target wards
- intervention intensity and coverage

Outputs:

- ranked intervention plan
- ward allocation
- estimated cooling impact
- cost breakdown
- candidate ranking table
- explainable rationale for selected actions

The current implementation uses a greedy baseline. It generates ward-intervention candidates through the scenario engine, ranks them by estimated cooling per rupee, selects at most one intervention per ward, and stops when the budget is exhausted.

This is intentionally hackathon-appropriate: simple, transparent, and easy to explain. The abstraction layer allows future replacement with integer programming, multi-objective optimization, equity constraints, or implementation feasibility scoring.

## 10. Dashboard Features

The Streamlit dashboard is the main review surface for judges and first-time users.

Pages:

- **Overview:** project summary, readiness indicators, hotspot preview, and executive signal
- **Hotspots Map:** filterable map of sampled land-surface heat intensity
- **Ward Rankings:** ward-level prioritization, choropleth map, KPI cards, and ranking table
- **Driver Explanations:** SHAP-backed global and ward-level explanation cards
- **Scenario Simulation:** interactive intervention simulation by ward, type, intensity, and coverage
- **Optimization:** budget-aware intervention allocation with cost assumptions and ranked outputs
- **AI Recommendations:** deterministic city briefing, ward summaries, and planner-ready recommendation table

See `docs/dashboard_walkthrough.md` for a page-by-page guide.

## 11. Repository Structure

```text
UrbanCool/
├── app.py
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── docs/
│   └── dashboard_walkthrough.md
├── notebooks/
├── outputs/
│   ├── baseline_xgboost_model.json
│   ├── shap_values.csv
│   ├── shap_beeswarm.png
│   ├── shap_global_importance.png
│   └── ward_heat_summary.geojson
├── src/
│   ├── dashboard/
│   ├── data_pipeline/
│   ├── explainability/
│   ├── feature_engineering/
│   ├── llm_layer/
│   ├── models/
│   ├── optimization/
│   └── scenario_engine/
└── tests/
```

## 12. Installation

Create and activate a virtual environment.

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Authenticate Google Earth Engine once:

```bash
earthengine authenticate --auth_mode=localhost
earthengine set_project urbancool-ai-500306
```

If using a different Earth Engine project, replace `urbancool-ai-500306` with your own project ID.

## 13. Running the Pipeline

Run the Earth Engine connection check:

```bash
python -m tests.test_gee_connection
```

Fetch and inspect the first feature summary:

```bash
python -m src.data_pipeline.gee_fetch
```

Export a sampled training table:

```bash
python -m src.data_pipeline.gee_fetch --export-csv data/processed/kolkata_training_sample.csv --sample-size 100000
```

Optionally include ECOSTRESS where coverage exists:

```bash
python -m src.data_pipeline.gee_fetch --include-ecostress --export-csv data/processed/kolkata_training_sample_ecostress.csv --sample-size 100000
```

Fetch OSM building and road layers:

```bash
python -m src.data_pipeline.osm_fetch
```

Optionally clean an externally downloaded CPCB CSV:

```bash
python -m src.data_pipeline.cpcb_ingest --input-csv data/external/cpcb_raw.csv
```

Train the baseline model:

```bash
python -m src.models.baseline_xgboost
```

Run SHAP attribution:

```bash
python -m src.explainability.shap_baseline
```

Train the experimental PINN:

```bash
python -m src.models.pinn
```

Export the full feature stack and run full-raster prediction:

```bash
python -m src.feature_engineering.full_raster_prediction
```

The full feature stack export is asynchronous through Earth Engine. After the task completes, download `feature_stack.tif` from the `UrbanCool_Exports` Google Drive folder, place it at `outputs/feature_stack.tif`, and run:

```bash
python -m src.feature_engineering.full_raster_prediction --skip-download
```

Generate ward-level heat summaries:

```bash
python -m src.feature_engineering.ward_aggregation
```

Expected key artifacts:

- `data/processed/kolkata_training_sample.csv`
- `outputs/baseline_xgboost_model.json`
- `outputs/shap_values.csv`
- `outputs/shap_beeswarm.png`
- `outputs/shap_global_importance.png`
- `outputs/predicted_lst_raster.tif`
- `outputs/ward_heat_summary.geojson`

## 14. Running the Dashboard

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

If using the local virtual environment directly:

```bash
venv/bin/streamlit run app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

The dashboard is artifact-first. If a data artifact is missing, the relevant page will show a readiness message instead of fabricating outputs.

## 15. Example Workflow

1. Open the dashboard and start on **Overview**.
2. Use **Hotspots Map** to inspect where high land-surface temperature samples cluster.
3. Open **Ward Rankings** to identify the highest-priority wards.
4. Use **Driver Explanations** to understand whether heat is associated with built density, vegetation loss, water absence, morphology, or radiation.
5. Open **Scenario Simulation** and test an intervention for a selected ward.
6. Open **Optimization**, set a budget, adjust cost assumptions, and generate a ranked intervention plan.
7. Open **AI Recommendations** to get planner-friendly summaries for the city and selected wards.
8. Use the recommendation table and optimization rationale as a presentation-ready action plan.

## 16. Limitations

UrbanCool AI is a hackathon prototype. The following limitations are documented tradeoffs:

- **Ward boundary completeness:** the current community ward dataset covers 141 of Kolkata's 144 wards.
- **Temporal scope:** features represent a March-June 2025 dry-season snapshot, not a seasonal or daily time series.
- **Scenario calibration:** intervention impacts are directional, bounded planning estimates, not calibrated CFD or field-measured effects.
- **PINN maturity:** the experimental PINN currently underperforms the XGBoost baseline on raw predictive fit.
- **CPCB integration:** station ingestion exists, but CPCB observations are not yet fused into the feature stack.
- **OSM fusion:** OSM building and road layers are fetched separately but not yet used in the core model features.
- **SVF approximation:** sky view factor and canyon geometry use raster-density proxies rather than full 3D viewshed calculations.
- **Water SHAP stability:** water's cooling direction is clear, but its exact global importance rank can shift because water bodies are spatially clustered.
- **Raster export:** the full feature stack exceeds Earth Engine's synchronous download cap and currently requires asynchronous Drive export.

These limitations are surfaced so reviewers can distinguish prototype scope from implementation gaps.

## 17. Future Work

Planned improvements:

- replace or augment the default scenario response model with a calibrated PINN
- add official KMC ward boundaries for full 144-ward coverage
- fuse OSM building footprints and roads into higher-resolution morphology features
- integrate CPCB station observations for atmospheric validation
- support seasonal comparisons and trend analysis
- add equity, feasibility, and implementation-readiness constraints to optimization
- calibrate intervention costs with local procurement or municipal planning data
- automate Google Drive export retrieval for full-raster prediction
- add downloadable reports for planners and reviewers
- package the dashboard for easier deployment
