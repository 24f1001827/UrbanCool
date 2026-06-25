#tests/test_gee_connection.py
import sys

import ee

from src.config import load_config
from src.data_pipeline.gee_auth import initialize_earth_engine


def main() -> None:
    config = load_config()
    ee_client = initialize_earth_engine(config.ee_project)

    image = ee_client.Image("USGS/SRTMGL1_003")
    image_type = image.getInfo()["type"]

    print(f"Connected to Earth Engine project: {config.ee_project}")
    print(f"Smoke-test image type: {image_type}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        sys.exit(1)

