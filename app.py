#app.py
from pathlib import Path

import pandas as pd
import streamlit as st


TRAINING_SAMPLE_PATH = Path("data/processed/kolkata_training_sample.csv")


def main() -> None:
    st.set_page_config(page_title="UrbanCool AI", layout="wide")
    st.title("UrbanCool AI")
    st.caption("Urban heat stress mapping and cooling intervention planning")

    if not TRAINING_SAMPLE_PATH.exists():
        st.info(
            "No sampled training table found yet. Run the data-pipeline export "
            "command from the README to generate the first model-ready dataset."
        )
        return

    data = pd.read_csv(TRAINING_SAMPLE_PATH)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Sampled locations", f"{len(data):,}")
    metric_cols[1].metric("Mean LST", f"{data['LST_C'].mean():.2f} C")
    metric_cols[2].metric("Mean NDVI", f"{data['NDVI'].mean():.3f}")
    metric_cols[3].metric("Mean built fraction", f"{data['BUILT_FRACTION'].mean():.3f}")

    st.subheader("Training Sample Preview")
    st.dataframe(data.head(100), use_container_width=True)

    st.subheader("Sample Locations")
    st.map(data[["latitude", "longitude"]].dropna())


if __name__ == "__main__":
    main()

