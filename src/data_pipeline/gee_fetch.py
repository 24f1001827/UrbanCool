#src/data_pipeline/gee_fetch.py

import argparse
import csv
import sys
from pathlib import Path
from pprint import pprint
from typing import Any
import ee
from src.config import ProjectConfig, load_config
from src.data_pipeline.gee_auth import initialize_earth_engine


LANDSAT_L2 = "LANDSAT/LC08/C02/T1_L2"
SENTINEL2_SR = "COPERNICUS/S2_SR_HARMONIZED"
DYNAMIC_WORLD = "GOOGLE/DYNAMICWORLD/V1"
KOLKATA_BOUNDARY = "FAO/GAUL/2015/level2"
GHSL_BUILT_HEIGHT_2018 = "JRC/GHSL/P2023A/GHS_BUILT_H/2018"
GHSL_BUILT_SURFACE_2020 = "JRC/GHSL/P2023A/GHS_BUILT_S/2020"
ERA5_LAND_DAILY = "ECMWF/ERA5_LAND/DAILY_AGGR"
ECOSTRESS_LSTE = "NASA/ECOSTRESS/L2T/LSTE"


def load_city_boundary(ee: Any, city_name: str):
    boundary = ee.FeatureCollection(KOLKATA_BOUNDARY).filter(
        ee.Filter.eq("ADM2_NAME", city_name)
    )
    count = boundary.size().getInfo()
    if count == 0:
        raise ValueError(f"No boundary found for city name: {city_name}")
    return boundary


def mask_landsat_l2_clouds(image: Any):
    qa = image.select("QA_PIXEL")
    cloud_shadow = qa.bitwiseAnd(1 << 3).eq(0)
    clouds = qa.bitwiseAnd(1 << 4).eq(0)
    return image.updateMask(cloud_shadow.And(clouds))


def build_landsat_feature_image(ee: Any, config: ProjectConfig, region: Any) -> Any:
    collection = (
        ee.ImageCollection(LANDSAT_L2)
        .filterBounds(region)
        .filterDate(config.start_date, config.end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", config.cloud_cover_max))
        .map(mask_landsat_l2_clouds)
    )

    image_count = collection.size().getInfo()
    if image_count == 0:
        raise ValueError(
            "No Landsat scenes matched the current filters. "
            "Try widening the date range or cloud-cover threshold."
        )

    composite = collection.median().clip(region)

    optical = (
        composite.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
        .multiply(0.0000275)
        .add(-0.2)
    )

    ndvi = optical.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    ndbi = optical.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")

    albedo = (
        optical.expression(
            "0.356 * blue + 0.130 * red + 0.373 * nir + 0.085 * swir1 + 0.072 * swir2 - 0.0018",
            {
                "blue": optical.select("SR_B2"),
                "red": optical.select("SR_B4"),
                "nir": optical.select("SR_B5"),
                "swir1": optical.select("SR_B6"),
                "swir2": optical.select("SR_B7"),
            },
        )
        .rename("ALBEDO")
    )

    lst_c = (
        composite.select("ST_B10")
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("LST_C")
    )

    feature_image = ee.Image.cat([ndvi, ndbi, albedo, lst_c]).clip(region)
    return feature_image


def mask_sentinel2_clouds(image: Any):
    scl = image.select("SCL")
    clear = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return image.updateMask(clear)


def build_sentinel_lulc_image(ee: Any, config: ProjectConfig, city: Any):
    collection = (
        ee.ImageCollection(SENTINEL2_SR)
        .filterBounds(city)
        .filterDate(config.start_date, config.end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", config.cloud_cover_max))
        .map(mask_sentinel2_clouds)
    )

    image_count = collection.size().getInfo()
    if image_count == 0:
        raise ValueError(
            "No Sentinel-2 scenes matched the current filters. "
            "Try widening the date range or cloud-cover threshold."
        )

    composite = collection.median().clip(city)
    reflectance = composite.select(["B2", "B3", "B4", "B8", "B11", "B12"]).divide(10_000)

    s2_ndvi = reflectance.normalizedDifference(["B8", "B4"]).rename("S2_NDVI")
    s2_ndbi = reflectance.normalizedDifference(["B11", "B8"]).rename("S2_NDBI")
    mndwi = reflectance.normalizedDifference(["B3", "B11"]).rename("S2_MNDWI")
    bsi = (
        reflectance.expression(
            "((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))",
            {
                "swir1": reflectance.select("B11"),
                "red": reflectance.select("B4"),
                "nir": reflectance.select("B8"),
                "blue": reflectance.select("B2"),
            },
        )
        .rename("S2_BSI")
    )

    dynamic_world = (
        ee.ImageCollection(DYNAMIC_WORLD)
        .filterBounds(city)
        .filterDate(config.start_date, config.end_date)
        .select(
            [
                "water",
                "trees",
                "grass",
                "flooded_vegetation",
                "crops",
                "shrub_and_scrub",
                "built",
                "bare",
                "snow_and_ice",
                "label",
            ]
        )
    )

    dw_count = dynamic_world.size().getInfo()
    if dw_count == 0:
        raise ValueError(
            "No Dynamic World scenes matched the current filters. "
            "Try widening the date range."
        )

    dw_probabilities = dynamic_world.select(
        [
            "water",
            "trees",
            "grass",
            "flooded_vegetation",
            "crops",
            "shrub_and_scrub",
            "built",
            "bare",
            "snow_and_ice",
        ]
    ).mean()
    dw_label = dynamic_world.select("label").mode().rename("DW_LABEL")

    dw_probabilities = dw_probabilities.rename(
        [
            "DW_WATER_PROB",
            "DW_TREES_PROB",
            "DW_GRASS_PROB",
            "DW_FLOODED_VEGETATION_PROB",
            "DW_CROPS_PROB",
            "DW_SHRUB_SCRUB_PROB",
            "DW_BUILT_PROB",
            "DW_BARE_PROB",
            "DW_SNOW_ICE_PROB",
        ]
    )

    return ee.Image.cat(
        [s2_ndvi, s2_ndbi, mndwi, bsi, dw_probabilities, dw_label]
    ).clip(city)


def build_morphology_image(ee: Any, city: Any):
    built_height = (
        ee.Image(GHSL_BUILT_HEIGHT_2018)
        .select("built_height")
        .rename("BUILDING_HEIGHT_M")
        .clip(city)
    )

    built_surface = ee.Image(GHSL_BUILT_SURFACE_2020).clip(city)
    built_fraction = (
        built_surface.select("built_surface")
        .divide(10_000)
        .clamp(0, 1)
        .rename("BUILT_FRACTION")
    )
    built_nres_fraction = (
        built_surface.select("built_surface_nres")
        .divide(10_000)
        .clamp(0, 1)
        .rename("BUILT_NRES_FRACTION")
    )

    morphology_mass_proxy = (
        built_height.multiply(built_fraction).rename("MORPHOLOGY_MASS_PROXY")
    )

    # --- Canyon aspect ratio (H/W) and Sky View Factor ---
    # Raster-based proxy in the absence of per-building footprint geometry.
    # H: focal-mean building height in a 150m neighborhood (captures local
    #    street-canyon scale rather than a single noisy pixel).
    # W: characteristic open-space spacing, derived from built_fraction —
    #    if a fraction f of the neighborhood is built, average building
    #    spacing scales as ~ pixel_size / sqrt(f). This is the standard
    #    morphological approximation used when footprint vectors aren't
    #    available (c.f. Oke's "urban canyon" idealization).
    kernel = ee.Kernel.circle(radius=150, units="meters")

    mean_height = built_height.reduceNeighborhood(
        reducer=ee.Reducer.mean(), kernel=kernel
    ).rename("MEAN_HEIGHT_150M")

    mean_built_fraction = built_fraction.reduceNeighborhood(
        reducer=ee.Reducer.mean(), kernel=kernel
    ).rename("MEAN_BUILT_FRACTION_150M")

    pixel_size_m = ee.Image.pixelArea().sqrt()
    spacing_w = (
        pixel_size_m.divide(mean_built_fraction.sqrt().max(0.05))
        .rename("SPACING_W_M")
    )

    canyon_aspect_ratio = (
        mean_height.divide(spacing_w.max(1))
        .rename("CANYON_ASPECT_RATIO_HW")
    )

    # Steyn (1980) empirical SVF for a symmetric urban canyon:
    # SVF = cos(arctan(H / (2W)))  -- bounded in [0, 1], 1 = fully open sky
    svf = (
        canyon_aspect_ratio.divide(2)
        .atan()
        .cos()
        .rename("SKY_VIEW_FACTOR")
    )

    return ee.Image.cat(
        [
            built_height,
            built_fraction,
            built_nres_fraction,
            morphology_mass_proxy,
            mean_height,
            canyon_aspect_ratio,
            svf,
        ]
    ).clip(city)


def build_meteorology_image(ee: Any, config: ProjectConfig, city: Any):
    collection = (
        ee.ImageCollection(ERA5_LAND_DAILY)
        .filterBounds(city)
        .filterDate(config.start_date, config.end_date)
    )

    image_count = collection.size().getInfo()
    if image_count == 0:
        raise ValueError(
            "No ERA5-Land scenes matched the current date range. "
            "Try an earlier end date if the requested period is too recent."
        )

    mean_weather = collection.select(
        [
            "temperature_2m",
            "dewpoint_temperature_2m",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m",
            "surface_pressure",
            "volumetric_soil_water_layer_1",
            "surface_net_solar_radiation_sum",
            "surface_net_thermal_radiation_sum",
            "surface_sensible_heat_flux_sum",
            "surface_latent_heat_flux_sum",
        ]
    ).mean()

    total_precipitation = (
        collection.select("total_precipitation_sum")
        .sum()
        .multiply(1000)
        .rename("PRECIPITATION_MM")
    )

    temp_c = mean_weather.select("temperature_2m").subtract(273.15).rename("AIR_TEMP_C")
    dewpoint_c = (
        mean_weather.select("dewpoint_temperature_2m")
        .subtract(273.15)
        .rename("DEWPOINT_C")
    )
    wind_speed = (
        mean_weather.select("u_component_of_wind_10m")
        .pow(2)
        .add(mean_weather.select("v_component_of_wind_10m").pow(2))
        .sqrt()
        .rename("WIND_SPEED_M_S")
    )
    pressure_hpa = mean_weather.select("surface_pressure").divide(100).rename(
        "SURFACE_PRESSURE_HPA"
    )
    soil_moisture = mean_weather.select("volumetric_soil_water_layer_1").rename(
        "SOIL_MOISTURE_L1"
    )
    net_solar_wm2 = mean_weather.select("surface_net_solar_radiation_sum").divide(
        86_400
    ).rename("NET_SOLAR_RADIATION_W_M2")
    net_thermal_wm2 = mean_weather.select("surface_net_thermal_radiation_sum").divide(
        86_400
    ).rename("NET_THERMAL_RADIATION_W_M2")
    sensible_heat_wm2 = mean_weather.select("surface_sensible_heat_flux_sum").divide(
        86_400
    ).rename("SENSIBLE_HEAT_FLUX_W_M2")
    latent_heat_wm2 = mean_weather.select("surface_latent_heat_flux_sum").divide(
        86_400
    ).rename("LATENT_HEAT_FLUX_W_M2")

    return ee.Image.cat(
        [
            temp_c,
            dewpoint_c,
            wind_speed,
            total_precipitation,
            pressure_hpa,
            soil_moisture,
            net_solar_wm2,
            net_thermal_wm2,
            sensible_heat_wm2,
            latent_heat_wm2,
        ]
    ).clip(city)


def build_ecostress_image(ee: Any, config: ProjectConfig, city: Any):
    collection = (
        ee.ImageCollection(ECOSTRESS_LSTE)
        .filterBounds(city)
        .filterDate(config.start_date, config.end_date)
    )

    image_count = collection.size().getInfo()
    if image_count == 0:
        raise ValueError(
            "No ECOSTRESS LST scenes matched this city/date window. "
            "Run without --include-ecostress or widen the date range."
        )

    ecostress_lst_c = (
        collection.select("LST")
        .median()
        .subtract(273.15)
        .rename("ECOSTRESS_LST_C")
        .clip(city)
    )
    return ecostress_lst_c


def build_feature_stack(ee: Any, config: ProjectConfig) -> tuple[Any, Any]:
    city = load_city_boundary(ee, config.city_name)
    return build_feature_stack_for_region(ee, config, city.geometry()), city


def build_feature_stack_for_region(ee: Any, config: ProjectConfig, region: Any) -> Any:
    landsat_features = build_landsat_feature_image(ee, config, region)
    sentinel_lulc_features = build_sentinel_lulc_image(ee, config, region)
    morphology_features = build_morphology_image(ee, region)
    meteorology_features = build_meteorology_image(ee, config, region)

    feature_images = [
        landsat_features,
        sentinel_lulc_features,
        morphology_features,
        meteorology_features,
    ]

    if config.include_ecostress:
        feature_images.append(build_ecostress_image(ee, config, region))

    return ee.Image.cat(feature_images)


def summarize_feature_image(
    ee: Any,
    feature_image: Any,
    city: Any,
    scale_m: int,
) -> dict:
    return feature_image.reduceRegion(
        reducer=ee.Reducer.minMax().combine(
            reducer2=ee.Reducer.mean(),
            sharedInputs=True,
        ),
        geometry=city.geometry(),
        scale=scale_m,
        bestEffort=True,
        maxPixels=1_000_000_000,
    ).getInfo()


def sample_feature_image(
    ee: Any,
    feature_image: Any,
    city: Any,
    scale_m: int,
    sample_size: int,
    seed: int,
) -> list[dict]:
    max_batch = 5000  # GEE getInfo() hard limit per FeatureCollection request
    rows: list[dict] = []
    remaining = sample_size
    batch_seed = seed

    while remaining > 0:
        batch_size = min(remaining, max_batch)

        samples = (
            feature_image.sample(
                region=city.geometry(),
                scale=scale_m,
                numPixels=batch_size,
                seed=batch_seed,
                geometries=True,
                tileScale=4,
            )
            .filter(ee.Filter.notNull(feature_image.bandNames()))
            .getInfo()
        )

        for feature in samples["features"]:
            properties = feature["properties"]
            longitude, latitude = feature["geometry"]["coordinates"]
            rows.append(
                {
                    "longitude": longitude,
                    "latitude": latitude,
                    **properties,
                }
            )

        print(f"  fetched batch seed={batch_seed}: {len(samples['features'])} rows (total so far: {len(rows)})")

        remaining -= batch_size
        batch_seed += 1  # different seed per batch so pixels aren't identical

    return rows


def write_rows_to_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        raise ValueError("No sampled rows were returned from Earth Engine.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["longitude", "latitude"] + sorted(
        key for row in rows for key in row.keys() if key not in {"longitude", "latitude"}
    )
    fieldnames = list(dict.fromkeys(fieldnames))

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    defaults = load_config()
    parser = argparse.ArgumentParser(description="Fetch first UrbanCool feature summary.")
    parser.add_argument("--project", default=defaults.ee_project)
    parser.add_argument("--city", default=defaults.city_name)
    parser.add_argument("--start-date", default=defaults.start_date)
    parser.add_argument("--end-date", default=defaults.end_date)
    parser.add_argument("--cloud-cover-max", type=int, default=defaults.cloud_cover_max)
    parser.add_argument("--scale", type=int, default=defaults.scale_m)
    parser.add_argument("--include-ecostress", action="store_true", default=defaults.include_ecostress)
    parser.add_argument("--export-csv", type=Path)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig(
        ee_project=args.project,
        city_name=args.city,
        start_date=args.start_date,
        end_date=args.end_date,
        cloud_cover_max=args.cloud_cover_max,
        scale_m=args.scale,
        include_ecostress=args.include_ecostress,
    )

    ee = initialize_earth_engine(config.ee_project)
    feature_image, city = build_feature_stack(ee, config)
    summary = summarize_feature_image(ee, feature_image, city, config.scale_m)

    print(
        f"UrbanCool feature summary for {config.city_name} "
        f"({config.start_date} to {config.end_date})"
    )
    pprint(summary)

    if args.export_csv:
        rows = sample_feature_image(
            ee=ee,
            feature_image=feature_image,
            city=city,
            scale_m=config.scale_m,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        write_rows_to_csv(rows, args.export_csv)
        print(f"Wrote {len(rows)} sampled training rows to {args.export_csv}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        sys.exit(1)
