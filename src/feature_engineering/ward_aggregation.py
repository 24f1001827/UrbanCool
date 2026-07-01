# src/feature_engineering/ward_aggregation.py

import argparse
import json
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
from xgboost import XGBRegressor

from src.config import load_config
from src.data_pipeline.gee_auth import initialize_earth_engine
from src.data_pipeline.gee_fetch import build_feature_stack_for_region
from src.models.baseline_xgboost import DROP_COLUMNS, TARGET_COLUMN


def fetch_ward_boundaries() -> gpd.GeoDataFrame:
    """
    Kolkata ward boundaries from the Datameet community Municipal_Spatial_Data
    repo (OSM has no usable KMC ward-level boundaries for Kolkata as of this
    check — see Known Limitations).
    """
    url = "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/master/Kolkata/kolkata.geojson"
    wards = gpd.read_file(url)
    return wards

def sample_full_grid(
    ee_client,
    feature_image,
    region,
    scale_m: int,
    batch_size: int = 5000,
) -> pd.DataFrame:
    """
    Pull a dense systematic grid covering the whole city, batched to respect
    GEE's getInfo limit. Uses a regular pixel grid rather than random sampling
    so coverage is complete, not just statistically representative.
    """
    pixel_count = feature_image.select(TARGET_COLUMN).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=region,
        scale=scale_m,
        bestEffort=True,
        maxPixels=1_000_000_000,
    ).getInfo()[TARGET_COLUMN]

    print(f"Estimated valid pixels at {scale_m}m: {pixel_count}")

    rows = []
    remaining = pixel_count
    seed = 0
    while remaining > 0:
        this_batch = min(remaining, batch_size)
        samples = (
            feature_image.sample(
                region=region,
                scale=scale_m,
                numPixels=this_batch,
                seed=seed,
                geometries=True,
                tileScale=4,
            )
            .filter(ee.Filter.notNull(feature_image.bandNames()))
            .getInfo()
        )
        for feature in samples["features"]:
            properties = feature["properties"]
            lon, lat = feature["geometry"]["coordinates"]
            rows.append({"longitude": lon, "latitude": lat, **properties})

        print(f"  batch seed={seed}: {len(samples['features'])} rows (total: {len(rows)})")
        remaining -= this_batch
        seed += 1

    return pd.DataFrame(rows)


def predict_lst(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    model = XGBRegressor()
    model.load_model(model_path)

    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]
    df = df.copy()
    df["PREDICTED_LST_C"] = model.predict(df[feature_cols])
    return df


def aggregate_to_wards(
    points_df: pd.DataFrame,
    wards_gdf: gpd.GeoDataFrame,
    agg_columns: list[str],
) -> gpd.GeoDataFrame:
    points_gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df["longitude"], points_df["latitude"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(points_gdf, wards_gdf, how="inner", predicate="within")

    ward_stats = joined.groupby("WARD")[agg_columns].mean().reset_index()
    ward_stats["pixel_count"] = joined.groupby("WARD").size().values

    result = wards_gdf.merge(ward_stats, on="WARD", how="left")
    return result


def parse_args() -> argparse.Namespace:
    defaults = load_config()
    parser = argparse.ArgumentParser(description="Aggregate predicted LST to ward level.")
    parser.add_argument("--model-path", type=Path, default=Path("outputs/baseline_xgboost_model.json"))
    parser.add_argument("--scale", type=int, default=defaults.scale_m)
    parser.add_argument("--output-geojson", type=Path, default=Path("outputs/ward_heat_summary.geojson"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    ee_client = initialize_earth_engine(config.ee_project)

    print("Fetching ward boundaries...")
    wards_gdf = fetch_ward_boundaries()
    print(f"Found {len(wards_gdf)} ward boundaries")

    # Build the sampling region from the actual ward union, not GAUL's
    # district boundary — GAUL's "Kolkata" polygon clips off southern KMC
    # wards (confirmed visually), so we sample exactly the area we report on.
    ward_union = wards_gdf.union_all()
    sampling_region = ee.Geometry(ward_union.__geo_interface__)

    feature_image = build_feature_stack_for_region(ee_client, config, sampling_region)
    # build_feature_stack still internally uses GAUL for masking/clipping
    # inside build_landsat_feature_image etc. — clip to ward_union instead:
    null_check = feature_image.mask().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=sampling_region,
        scale=30,
        bestEffort=True,
        maxPixels=1e9
    ).getInfo()
    print("Per-band coverage (1.0 = fully covered, lower = gaps):")
    for band, coverage in sorted(null_check.items(), key=lambda x: x[1]):
        print(f"  {band}: {coverage:.3f}")
    print("Sampling full city grid (this may take a while)...")
    points_df = sample_full_grid(ee_client, feature_image, sampling_region, args.scale)

    print("Running model predictions...")
    points_df = predict_lst(points_df, args.model_path)

    agg_columns = ["PREDICTED_LST_C", "NDVI", "NDBI", "BUILT_FRACTION", "DW_WATER_PROB"]
    print("Aggregating to ward level...")
    ward_summary = aggregate_to_wards(points_df, wards_gdf, agg_columns)

    args.output_geojson.parent.mkdir(parents=True, exist_ok=True)
    ward_summary.to_file(args.output_geojson, driver="GeoJSON")
    print(f"Saved ward-level heat summary to {args.output_geojson}")

    print("\nWard summary preview:")
    print(ward_summary[["WARD", "PREDICTED_LST_C", "pixel_count"]].sort_values(
        "PREDICTED_LST_C", ascending=False
    ).head(10))


if __name__ == "__main__":
    main()