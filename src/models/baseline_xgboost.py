# src/models/baseline_xgboost.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


TARGET_COLUMN = "LST_C"
DROP_COLUMNS = [
    "longitude", "latitude", "LST_C", "DW_LABEL", "ECOSTRESS_LST_C",
    "AIR_TEMP_C", "WIND_SPEED_M_S", "SOIL_MOISTURE_L1",
    "SURFACE_PRESSURE_HPA", "DEWPOINT_C",
    "SENSIBLE_HEAT_FLUX_W_M2",
    "LATENT_HEAT_FLUX_W_M2",
    "NET_THERMAL_RADIATION_W_M2",
]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TARGET_COLUMN])
    return df


def spatial_train_val_test_split(
    df: pd.DataFrame,
    n_lat_bins: int = 10,
    n_lon_bins: int = 10,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    # Bin the city into a lat/lon grid, then split whole GRID CELLS across
    # train/val/test — not individual points. This avoids leakage from
    # spatially autocorrelated neighboring pixels ending up split across sets.
    rng = np.random.default_rng(seed)

    lat_bin = pd.cut(df["latitude"], bins=n_lat_bins, labels=False)
    lon_bin = pd.cut(df["longitude"], bins=n_lon_bins, labels=False)
    df = df.copy()
    df["_grid_cell"] = lat_bin.astype(str) + "_" + lon_bin.astype(str)

    unique_cells = np.array(df["_grid_cell"].unique().tolist())
    rng.shuffle(unique_cells)

    n_cells = len(unique_cells)
    n_test = max(1, int(n_cells * test_frac))
    n_val = max(1, int(n_cells * val_frac))

    test_cells = set(unique_cells[:n_test])
    val_cells = set(unique_cells[n_test:n_test + n_val])
    train_cells = set(unique_cells[n_test + n_val:])

    train_df = df[df["_grid_cell"].isin(train_cells)].drop(columns=["_grid_cell"])
    val_df = df[df["_grid_cell"].isin(val_cells)].drop(columns=["_grid_cell"])
    test_df = df[df["_grid_cell"].isin(test_cells)].drop(columns=["_grid_cell"])

    return train_df, val_df, test_df


def make_xy(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    return X, y, feature_cols


def evaluate(model, X, y, split_name: str):
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds) ** 0.5
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"[{split_name}] RMSE={rmse:.3f} C  MAE={mae:.3f} C  R2={r2:.3f}")
    return rmse, mae, r2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline XGBoost LST model.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/processed/kolkata_training_sample.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/baseline_xgboost_model.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input_csv)

    train_df, val_df, test_df = spatial_train_val_test_split(df, seed=args.seed)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    X_train, y_train, feature_cols = make_xy(train_df)
    X_val, y_val, _ = make_xy(val_df)
    X_test, y_test, _ = make_xy(test_df)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val, y_val, "val")
    evaluate(model, X_test, y_test, "test")

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 feature importances:")
    print(importances.head(10))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model_out)
    print(f"\nSaved model to {args.model_out}")


if __name__ == "__main__":
    main()