#src/data_pipeline/osm_fetch.py

import argparse
from pathlib import Path

import geopandas as gpd
import osmnx as ox

from src.config import load_config


def fetch_osm_layers(place_name: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    buildings = ox.features_from_place(place_name, tags={"building": True})
    roads = ox.graph_to_gdfs(
        ox.graph_from_place(place_name, network_type="drive"),
        nodes=False,
        edges=True,
    )
    return buildings.reset_index(), roads.reset_index()


def write_layer(layer: gpd.GeoDataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    layer.to_file(temp_path, driver="GeoJSON")
    temp_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser(description="Fetch OSM building and road layers.")
    parser.add_argument("--place", default=f"{config.city_name}, India")
    parser.add_argument("--buildings-out", type=Path, default=Path("data/external/osm_buildings.geojson"))
    parser.add_argument("--roads-out", type=Path, default=Path("data/external/osm_roads.geojson"))
    parser.add_argument("--buildings-only", action="store_true")
    parser.add_argument("--roads-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.buildings_only and args.roads_only:
        raise ValueError("Choose either --buildings-only or --roads-only, not both.")

    if not args.roads_only:
        buildings = ox.features_from_place(args.place, tags={"building": True}).reset_index()
        write_layer(buildings, args.buildings_out)
        print(f"Wrote {len(buildings):,} OSM building features to {args.buildings_out}")

    if not args.buildings_only:
        roads = ox.graph_to_gdfs(
            ox.graph_from_place(args.place, network_type="drive"),
            nodes=False,
            edges=True,
        ).reset_index()
        write_layer(roads, args.roads_out)
        print(f"Wrote {len(roads):,} OSM road features to {args.roads_out}")


if __name__ == "__main__":
    main()
