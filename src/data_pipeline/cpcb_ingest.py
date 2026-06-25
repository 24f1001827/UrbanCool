#src/data_pipeline/cpcb_ingest.py

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"station", "datetime", "latitude", "longitude"}


def load_cpcb_csv(input_path: Path) -> pd.DataFrame:
    data = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "CPCB CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    data = data.dropna(subset=["datetime", "latitude", "longitude"])
    return data


def aggregate_station_daily(data: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = data.select_dtypes(include="number").columns.difference(
        ["latitude", "longitude"]
    )
    grouped = (
        data.assign(date=data["datetime"].dt.date)
        .groupby(["station", "date", "latitude", "longitude"], as_index=False)[numeric_columns]
        .mean()
    )
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and aggregate CPCB station CSV data.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/external/cpcb_station_daily.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_cpcb_csv(args.input_csv)
    daily = aggregate_station_daily(data)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(daily):,} CPCB station-day rows to {args.output_csv}")


if __name__ == "__main__":
    main()

