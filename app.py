import streamlit as st

from src.dashboard import DashboardArtifacts, render_app


def main() -> None:
    st.set_page_config(page_title="UrbanCool AI", layout="wide")
    render_app(DashboardArtifacts())


if __name__ == "__main__":
    main()
