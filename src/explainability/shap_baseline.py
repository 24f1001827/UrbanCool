# src/explainability/shap_baseline.py

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from xgboost import XGBRegressor

from src.models.baseline_xgboost import DROP_COLUMNS, TARGET_COLUMN, load_dataset, make_xy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP attribution on the baseline model.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/processed/kolkata_training_sample.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/baseline_xgboost_model.json"))
    parser.add_argument("--sample-size", type=int, default=3000, help="Rows to use for SHAP (full set is slow)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_dataset(args.input_csv)
    if len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42)

    X, y, feature_cols = make_xy(df)

    model = XGBRegressor()
    model.load_model(args.model_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Global driver ranking (bar plot)
    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(args.output_dir / "shap_global_importance.png", dpi=150)
    plt.close()

    # Beeswarm: direction + magnitude per feature
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(args.output_dir / "shap_beeswarm.png", dpi=150)
    plt.close()

    # Save per-row SHAP values as CSV for dashboard use later
    shap_df = pd.DataFrame(shap_values.values, columns=feature_cols)
    shap_df["base_value"] = shap_values.base_values
    shap_df["prediction"] = model.predict(X)
    shap_df[TARGET_COLUMN] = y.values
    shap_df.to_csv(args.output_dir / "shap_values.csv", index=False)

    mean_abs_shap = shap_df[feature_cols].abs().mean().sort_values(ascending=False)
    print("Mean |SHAP value| per feature (top 10):")
    print(mean_abs_shap.head(10))

    print(f"\nSaved plots and values to {args.output_dir}")


if __name__ == "__main__":
    main()
