"""
Application configuration.

All configuration is read from environment variables (via a .env file in
development). Nothing here is hard-coded per the project's coding rules —
every path, threshold, and credential-adjacent value comes from Settings.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object, populated from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- App ---
    app_name: str = "IBVAP"
    app_env: str = "development"
    debug: bool = True

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Database ---
    database_url: str = "sqlite:///./data/ibvap.db"

    # --- Storage ---
    storage_path: str = "./data"

    # --- AI models (used from Phase 3 onward) ---
    yolo_model: str = "models/yolo/yolo11n.pt"
    confidence_threshold: float = 0.45
    process_fps: int = 10

    # --- Night detection (Phase 5) ---
    night_start: str = "19:00"
    night_end: str = "06:00"
    night_brightness_threshold: int = 60

    # --- Activity engine (Phase 5) ---
    loitering_seconds: int = 30

    # --- ANPR (Phase 6) ---
    anpr_min_vehicle_width: int = 80
    anpr_min_vehicle_height: int = 60
    anpr_min_confidence: float = 0.3

    # --- Logging ---
    log_level: str = "INFO"

    @property
    def storage_dir(self) -> Path:
        """Resolved, guaranteed-to-exist storage root."""
        path = Path(self.storage_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def evidence_dir(self) -> Path:
        path = self.storage_dir / "evidence"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def videos_dir(self) -> Path:
        path = self.storage_dir / "videos"
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor — import and call this, don't instantiate Settings() directly."""
    return Settings()
