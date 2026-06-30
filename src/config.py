"""Centralized, typed application settings for ZeroVRAM-AirLLM-Engine."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables / `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    hugging_face_token: SecretStr

    project_root: Path = PROJECT_ROOT
    models_dir: Path = PROJECT_ROOT / "models"
    logs_dir: Path = PROJECT_ROOT / "logs"
    results_dir: Path = PROJECT_ROOT / "results"

    @property
    def hf_cache_dir(self) -> Path:
        """Hugging Face model weight cache, kept under `models_dir`."""
        return self.models_dir / "hf_cache"

    def ensure_directories(self) -> None:
        """Create the runtime data directories if they do not already exist."""
        for directory in (self.models_dir, self.logs_dir, self.results_dir, self.hf_cache_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, process-wide `Settings` instance."""
    return Settings()
