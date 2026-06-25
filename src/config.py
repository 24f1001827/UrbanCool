import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    ee_project: str = os.getenv("URBANCOOL_EE_PROJECT", "urbancool-ai-500306")
    city_name: str = os.getenv("URBANCOOL_CITY", "Kolkata")
    start_date: str = os.getenv("URBANCOOL_START_DATE", "2025-03-01")
    end_date: str = os.getenv("URBANCOOL_END_DATE", "2025-06-01")
    cloud_cover_max: int = int(os.getenv("URBANCOOL_CLOUD_COVER_MAX", "20"))
    scale_m: int = int(os.getenv("URBANCOOL_SCALE_M", "30"))
    include_ecostress: bool = os.getenv("URBANCOOL_INCLUDE_ECOSTRESS", "0") == "1"


def load_config() -> ProjectConfig:
    return ProjectConfig()
