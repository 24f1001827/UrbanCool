#src/data_pipeline/gee_auth.py

def initialize_earth_engine(project: str):
    try:
        import ee
    except ImportError as exc:
        message = (
            "Could not import the Earth Engine Python package. If the error mentions "
            "cryptography or a DLL, reinstall the environment packages or run the "
            "script from your normal local terminal where the virtual environment has "
            "permission to load native libraries."
        )
        raise RuntimeError(message) from exc

    try:
        ee.Initialize(project=project)
    except Exception as exc:
        message = (
            "Earth Engine is not initialized. Run this once in your activated "
            "environment: earthengine authenticate --auth_mode=localhost, then "
            f"earthengine set_project {project}."
        )
        raise RuntimeError(message) from exc

    return ee
