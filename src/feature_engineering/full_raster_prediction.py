# src/feature_engineering/full_raster_prediction.py

import argparse
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from xgboost import XGBRegressor

from src.config import load_config
from src.data_pipeline.gee_auth import initialize_earth_engine
from src.data_pipeline.gee_fetch import build_feature_stack_for_region
from src.models.baseline_xgboost import DROP_COLUMNS, TARGET_COLUMN


WARDS_URL = "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/master/Kolkata/kolkata.geojson"


def get_sampling_region(ee_client) -> tuple[ee.Geometry, gpd.GeoDataFrame]:
    wards_gdf = gpd.read_file(WARDS_URL)
    ward_union = wards_gdf.union_all()
    region = ee.Geometry(ward_union.__geo_interface__)
    return region, wards_gdf


def export_feature_stack_geotiff(
    ee_client,
    feature_image,
    region,
    scale_m: int,
    output_path: Path,
    drive_folder: str = "UrbanCool_Exports",
):
    """
    Exports via Earth Engine batch task to Google Drive — required because
    direct getDownloadURL has a 50MB hard cap, and the full feature stack
    exceeds that at city scale. This is asynchronous: the function submits
    the task and polls until it completes, then you must manually download
    the file from Drive (or use the Drive API to pull it automatically).
    """
    task = ee_client.batch.Export.image.toDrive(
        image=feature_image.toFloat(),
        description="urbancool_feature_stack",
        folder=drive_folder,
        fileNamePrefix="feature_stack",
        region=region,
        scale=scale_m,
        crs="EPSG:4326",
        maxPixels=1_000_000_000,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"Export task started: {task.id}")
    print("Polling for completion (this can take several minutes)...")

    import time
    while task.active():
        time.sleep(15)
        status = task.status()
        print(f"  status: {status['state']}")

    final_status = task.status()
    if final_status["state"] != "COMPLETED":
        raise RuntimeError(f"Export failed: {final_status}")

    print(f"\nExport complete. File is in your Google Drive under '{drive_folder}/feature_stack.tif'")
    print(f"Download it manually from Drive, then place it at: {output_path}")
    print("Once downloaded, rerun this script with --skip-download to run prediction on it.")

def run_pixelwise_prediction(
    feature_tiff_path: Path,
    model_path: Path,
    output_path: Path,
):
    with rasterio.open(feature_tiff_path) as src:
        band_names = list(src.descriptions)
        data = src.read()  # shape: (bands, height, width)
        profile = src.profile

    print(f"Loaded raster: {data.shape[0]} bands, {data.shape[1]}x{data.shape[2]} pixels")
    print(f"Band order: {band_names}")

    model = XGBRegressor()
    model.load_model(model_path)
    expected_features = model.get_booster().feature_names

    missing = [f for f in expected_features if f not in band_names]
    if missing:
        raise ValueError(f"Bands missing from exported raster: {missing}")

    band_index = {name: i for i, name in enumerate(band_names)}
    n_bands, height, width = data.shape

    # Reshape to (pixels, features) in the exact order the model expects
    feature_stack = np.stack(
        [data[band_index[f]].ravel() for f in expected_features], axis=1
    )

    valid_mask = ~np.isnan(feature_stack).any(axis=1)
    predictions = np.full(feature_stack.shape[0], np.nan, dtype=np.float32)

    if valid_mask.sum() > 0:
        predictions[valid_mask] = model.predict(feature_stack[valid_mask])

    predicted_raster = predictions.reshape(height, width)

    output_profile = profile.copy()
    output_profile.update(count=1, dtype="float32", nodata=np.nan)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(predicted_raster, 1)
        dst.set_band_description(1, "PREDICTED_LST_C")

    print(f"Saved predicted LST raster to {output_path}")
    print(f"Valid pixels: {valid_mask.sum()} / {len(valid_mask)}")
    print(f"Predicted LST range: {np.nanmin(predicted_raster):.2f} - {np.nanmax(predicted_raster):.2f} C")


def parse_args() -> argparse.Namespace:
    defaults = load_config()
    parser = argparse.ArgumentParser(description="Predict LST across the full city raster.")
    parser.add_argument("--model-path", type=Path, default=Path("outputs/baseline_xgboost_model.json"))
    parser.add_argument("--scale", type=int, default=defaults.scale_m)
    parser.add_argument("--feature-tiff", type=Path, default=Path("outputs/feature_stack.tif"))
    parser.add_argument("--output-tiff", type=Path, default=Path("outputs/predicted_lst_raster.tif"))
    parser.add_argument("--skip-download", action="store_true", help="Reuse an existing feature_stack.tif")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    if not args.skip_download:
        ee_client = initialize_earth_engine(config.ee_project)
        region, _ = get_sampling_region(ee_client)
        feature_image = build_feature_stack_for_region(ee_client, config, region)
        export_feature_stack_geotiff(ee_client, feature_image, region, args.scale, args.feature_tiff)

    run_pixelwise_prediction(args.feature_tiff, args.model_path, args.output_tiff)


if __name__ == "__main__":
    main()