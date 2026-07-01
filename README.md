# UrbanCool AI

UrbanCool AI is a hackathon prototype for urban heat stress mapping and cooling intervention planning.

This repository has a working data pipeline, a validated baseline LST model, SHAP-based driver attribution, and an experimental physics-informed neural network (PINN). Next milestone is ward-level zonal aggregation for the dashboard.

## Setup

1. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Make sure Earth Engine authentication has been completed once:

```powershell
earthengine authenticate --auth_mode=localhost
earthengine set_project urbancool-ai-500306
```

3. Run the connection check:

```powershell
python -m tests.test_gee_connection
```

4. Run the first data-pipeline summary:

```powershell
python -m src.data_pipeline.gee_fetch
```

5. Export a sampled training table (batched internally to work around GEE's 5,000-element getInfo limit):

```powershell
python -m src.data_pipeline.gee_fetch --export-csv data/processed/kolkata_training_sample.csv --sample-size 100000
```

6. Optionally include ECOSTRESS LST where coverage exists:

```powershell
python -m src.data_pipeline.gee_fetch --include-ecostress --export-csv data/processed/kolkata_training_sample_ecostress.csv --sample-size 100000
```

7. Fetch OSM building and road layers:

```powershell
python -m src.data_pipeline.osm_fetch
```

8. Clean a downloaded CPCB station CSV (optional — not currently used in the feature stack; see Known Limitations):

```powershell
python -m src.data_pipeline.cpcb_ingest --input-csv data/external/cpcb_raw.csv
```

9. Train the baseline XGBoost LST model:

```powershell
python -m src.models.baseline_xgboost
```

10. Train the experimental physics-informed neural network (currently underperforms the baseline on raw fit; see Known Limitations):

```powershell
python -m src.models.pinn
```

11. Run SHAP driver attribution on the trained baseline model:

```powershell
python -m src.explainability.shap_baseline
```

12. Export the full feature stack as a GeoTIFF via Earth Engine batch export (asynchronous — submits a task and polls until completion, then must be manually downloaded from Google Drive):

```powershell
python -m src.feature_engineering.full_raster_prediction
```

   Download `feature_stack.tif` from the `UrbanCool_Exports` folder in Google Drive and place it at `outputs/feature_stack.tif`, then run pixel-wise prediction locally:

```powershell
python -m src.feature_engineering.full_raster_prediction --skip-download
```

13. Run the ward-level aggregation (predicts across the full city grid and aggregates to ward polygons):

```powershell
python -m src.feature_engineering.ward_aggregation
```

## Current Milestone

The current pipeline:

- initializes Google Earth Engine without repeatedly opening browser authentication
- loads the Kolkata district boundary
- builds a Landsat 8 Level-2 median composite
- computes NDVI, NDBI, land surface temperature, and a simple albedo proxy
- adds Sentinel-2 surface-reflectance indices and Dynamic World land-cover probabilities/classes
- adds GHSL urban morphology features: building height (2018), built-up fraction (2020, nearest available epoch — GHSL has no 2018 BUILT_S product), non-residential built-up fraction, a height-density mass proxy, focal-mean building height, canyon aspect ratio (H/W), and a Sky View Factor approximation (Steyn 1980 formula, raster-density based — see Known Limitations)
- adds ERA5-Land meteorological features: 2m air temperature, dewpoint, wind speed, precipitation, surface pressure, soil moisture, and radiation/heat-flux context
- optionally adds ECOSTRESS LST as a higher-resolution thermal source when useful coverage exists
- includes separate OSM acquisition for building footprints and road geometry (not yet fused into the GEE feature stack)
- includes CPCB station CSV ingestion for externally downloaded station observations (not yet wired into the pipeline — deferred, see below)
- exports up to 100,000+ sampled rows for model training, batched to respect GEE's getInfo limits

**Baseline model (`src/models/baseline_xgboost.py`):** XGBoost regressor predicting `LST_C`, spatial train/val/test split (10×10 lat/lon grid cells, split at the cell level to avoid pixel-adjacency leakage).

Current performance (100K-row sample): train R²=0.891, val R²=0.775, test R²=0.745 (RMSE ≈ 0.8–1.0°C).

Top SHAP drivers, in order of mean |SHAP value|: `BUILT_NRES_FRACTION`, `NDBI`, `DW_WATER_PROB`, `MEAN_HEIGHT_150M`, `NET_SOLAR_RADIATION_W_M2`, `NDVI`. Directionally consistent with physical expectations — water and vegetation cool, built-up density and non-residential land use heat. Water's exact importance *rank* is somewhat unstable across spatial folds due to spatial clustering of water bodies (see Known Limitations), though its cooling *direction* is unambiguous in SHAP.

**Experimental PINN (`src/models/pinn.py`):** feedforward NN with a surface-energy-balance physics residual term (Rn = H + LE + G) added to the loss. Currently underperforms the XGBoost baseline on raw fit (test R²=0.690 vs 0.745). Retained for possible use in the scenario engine, where physically-constrained extrapolation under interventions may matter more than raw fit — decision deferred until the scenario engine is built.

**Ward-level aggregation (`src/feature_engineering/ward_aggregation.py`):** applies the trained baseline model across a full-resolution city-wide pixel grid (sampled over the union of all ward polygons, not the GAUL district boundary — see Known Limitations history), then aggregates predictions to ward level via spatial join. Produces `outputs/ward_heat_summary.geojson` with mean predicted LST, NDVI, NDBI, built fraction, and water probability per ward. Covers 141 of Kolkata's 144 wards (see Known Limitations). Output is ready for dashboard rendering (clickable ward map).

**Full-resolution heat map (`src/feature_engineering/full_raster_prediction.py`):** exports the complete feature stack as a multi-band GeoTIFF via Earth Engine batch export (direct download isn't possible at this size — exceeds GEE's 50MB synchronous download cap), then runs the trained XGBoost model pixel-by-pixel locally to produce a continuous predicted-LST raster (`outputs/predicted_lst_raster.tif`) covering all of Kolkata at 30m resolution. Output range: 29.45–42.44°C, consistent with training data range. Visual inspection confirms water bodies and green pockets correctly predicted as cooler, dense built-up core as hottest — consistent with SHAP driver attribution. This is the primary "Heat Stress Map" deliverable named in the official problem statement.

## Known Limitations & Roadmap

This section documents deliberate scoping decisions made to hit the 15-day hackathon
timeline. Each item below is a conscious tradeoff, not an oversight.

| Component | Current MVP Approach | Full / Future Approach | Why Deferred |
|---|---|---|---|
| Sky View Factor (SVF) & canyon aspect ratio | Raster-density proxy from GHSL building height + built-fraction rasters, using the Steyn (1980) empirical canyon formula | True geometric SVF: OSM building footprint polygons extruded by GHSL height into a 3D city model, with per-point hemispherical viewshed calculation | Footprint-level viewshed computation is a multi-day engineering task on its own; raster proxy captures the same density/height signal at a fraction of the cost |
| GHSL epoch alignment | Built height (2018) and built surface/fraction (2020) — nearest available epochs; GHSL does not ship a 2018 BUILT_S product | True multi-temporal alignment, or multi-temporal GHSL series to track morphology change over years | GHSL release schedule constraint, not a pipeline bug |
| Atmospheric ground-truthing | ERA5-Land reanalysis only; several ERA5 fields (air temp, wind speed, soil moisture, surface pressure, dewpoint) dropped from model features after confirming near-zero spatial variance across Kolkata at ERA5's native ~9km resolution | ERA5 cross-validated against CPCB CAAQMS ground station readings (temperature, humidity, wind) at finer spatial resolution | CPCB data requires manual, weekly-chunked export from the CCR portal (no clean API) and spatial interpolation to match raster grid — deferred as a stretch goal, not required for model to function |
| CPCB data pipeline | Ingestion script exists (`cpcb_ingest.py`) but not wired into the feature stack | One-time historical batch pull for the training window (not a live/continuous feed — not needed for this use case) | Time better spent on baseline model + physics-informed layer first |
| OSM building/road layers | Fetched separately (`osm_fetch.py`), not yet fused into the GEE feature stack | Fuse OSM footprint geometry into morphology features for higher-precision SVF/canyon metrics | Deferred until SHAP/dashboard work shows morphology precision is a real bottleneck, not a guess |
| Rare land-cover classes (flooded vegetation, shrub/scrub) | Very few training examples even at 100K total samples (flooded_vegetation: 21, shrub_scrub: 114) — model predictions in these specific zones should be treated with low confidence | Targeted/stratified sampling to oversample rare classes specifically | These are genuinely rare land-cover types in Kolkata; brute-force sampling more rows doesn't meaningfully fix rarity |
| Ward boundary completeness | Datameet community ward dataset covers 141 of KMC's 144 official wards (wards 142–144 absent from source) | Official KMC ward shapefile, if obtainable, for full 144-ward coverage | Datameet source predates the most recent ward delimitation; 141/144 (~98%) coverage is sufficient for MVP demonstration |
| Water-driver SHAP stability | `DW_WATER_PROB`'s cooling *direction* is unambiguous and strong in SHAP, but its relative importance *rank* shifts across different spatial train/val/test splits, due to water bodies being spatially clustered (rivers/ponds) rather than evenly distributed across the city grid | N/A — inherent to spatial clustering of water features | Documented as an interpretation caveat, not a bug to fix |
| Physics-informed NN (PINN) | Implemented with an energy-balance residual loss term; currently underperforms XGBoost baseline on raw fit (test R²=0.690 vs 0.745) | Tune `lambda_physics`, physics-head formulation, or use PINN selectively for scenario-engine extrapolation rather than the main hotspot model | Decision on whether PINN is needed deferred until scenario engine requirements are clearer |
| Data temporality | All raster features are a single March–June 2025 median/mean composite ("representative dry-season snapshot"), not a time series — no day-to-day or seasonal change is captured | Multiple seasonal composites for comparison (dry season vs monsoon) | Not required for spatial hotspot identification and intervention placement, which is the core ask of the problem statement |
| Raster export method | City-wide feature stack exported via asynchronous Earth Engine batch task to Google Drive, then manually downloaded | Automated download via Google Drive API, or server-side prediction inside Earth Engine (not possible — XGBoost models aren't natively executable in GEE) | Direct synchronous download (`getDownloadURL`) is capped at 50MB; the full 35-band feature stack exceeds this at city scale |
## Repository Structure

```text
UrbanCool/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   ├── data_pipeline/
│   ├── feature_engineering/
│   ├── models/
│   ├── explainability/
│   ├── scenario_engine/
│   ├── optimization/
│   ├── dashboard/
│   └── llm_layer/
├── configs/
├── tests/
├── outputs/
├── app.py
├── requirements.txt
└── README.md
```

## Next Milestones

- build the Streamlit/folium dashboard with clickable ward regions
- build scenario simulation module (intervention effects on LST)
- build constrained optimization module (budget-allocated intervention placement)
- decide whether PINN is needed for scenario-engine physical consistency