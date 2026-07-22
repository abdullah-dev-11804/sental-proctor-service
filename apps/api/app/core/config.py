from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    api_shared_secret: str = "dev-secret-change-me"
    cors_origins_raw: str = Field(default="*", alias="CORS_ORIGINS")

    local_storage_root: Path = Path("./storage")

    identity_pass_threshold: float = 0.72
    identity_review_threshold: float = 0.52
    identity_min_brightness: float = 35.0
    identity_min_blur: float = 35.0

    redis_url: str = "redis://localhost:6379/0"

    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "sental-minio"
    s3_secret_key: str = "change-me-minio-secret"
    s3_bucket_temp: str = "proctoring-temp"
    s3_bucket_evidence: str = "proctoring-evidence"
    s3_bucket_reports: str = "proctoring-reports"

    livekit_url: str = "wss://livekit.example.kz"
    livekit_api_key: str = "dev-livekit-key"
    livekit_api_secret: str = "dev-livekit-secret"

    moodle_webhook_url: str = ""
    moodle_webhook_secret: str = ""

    @property
    def cors_origins(self) -> list[str]:
        value = self.cors_origins_raw.strip()
        if value == "*":
            return ["*"]
        return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.local_storage_root.mkdir(parents=True, exist_ok=True)
    return settings
