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
    storage_backend: str = "local"
    storage_require_ready: bool = False
    reference_encryption_key_file: str = ""

    identity_pass_threshold: float = 0.85
    identity_review_threshold: float = 0.70
    identity_min_brightness: float = 35.0
    identity_min_blur: float = 35.0
    identity_engine: str = "scrfd_adaface"
    identity_model_root: Path = Path("./models")
    identity_scrfd_model: str = "scrfd_2.5g_kps.onnx"
    identity_adaface_model: str = "adaface_ir50_ms1mv2.onnx"
    identity_scrfd_input_size: int = 640
    identity_adaface_input_size: int = 160
    identity_adaface_color_order: str = "bgr"
    identity_min_face_confidence: float = 0.65
    identity_min_face_width_ratio: float = 0.16
    identity_max_face_width_ratio: float = 0.62
    identity_center_tolerance_x: float = 0.22
    identity_center_tolerance_y: float = 0.28
    identity_require_active_liveness: bool = False
    identity_require_enrollment_liveness: bool = False
    identity_min_enrollment_frames: int = 3
    identity_max_enrollment_frames: int = 12
    identity_min_template_consistency: float = 0.35
    identity_enrollment_min_brightness: float = 45.0
    identity_enrollment_min_blur: float = 45.0
    identity_enrollment_min_face_confidence: float = 0.75
    identity_enrollment_min_face_width_ratio: float = 0.18
    identity_enrollment_max_face_width_ratio: float = 0.58
    identity_enrollment_center_tolerance_x: float = 0.18
    identity_enrollment_center_tolerance_y: float = 0.23
    identity_min_live_frames: int = 1
    identity_enable_debug_capture: bool = False
    identity_debug_root: Path = Path("./storage/debug")
    identity_min_liveness_yaw_delta: float = 0.16
    identity_min_liveness_embedding_delta: float = 0.012
    identity_min_liveness_samples: int = 2
    identity_require_passive_antispoof: bool = False
    identity_temporal_passive_pad_enabled: bool = False
    identity_passive_min_valid_frames: int = 5
    identity_passive_capture_window_ms: int = 4500
    identity_antispoof_model: str = ""
    identity_antispoof_input_size: int = 80
    # Upstream Silent-Face-Anti-Spoofing MiniFASNet models use class 1 for a real face.
    identity_antispoof_live_class_index: int = 1
    identity_antispoof_crop_scale: float = 2.7
    identity_antispoof_threshold: float = 0.78
    identity_antispoof_spoof_threshold: float = 0.35
    identity_headpose_model: str = ""
    identity_require_headpose_liveness: bool = False
    identity_active_liveness_enabled: bool = False
    identity_headpose_challenge_enabled: bool = False
    identity_illumination_challenge_enabled: bool = False
    identity_liveness_challenge_ttl_seconds: int = 45
    identity_liveness_challenge_timeout_ms: int = 9000
    identity_liveness_retry_limit: int = 3
    identity_liveness_retry_window_seconds: int = 900
    identity_liveness_max_invalid_frame_ratio: float = 0.45
    identity_headpose_turn_degrees: float = 14.0
    identity_headpose_center_degrees: float = 9.0
    identity_headpose_min_hold_frames: int = 2
    identity_headpose_min_progress_degrees: float = 7.0
    identity_headpose_left_sign: int = 1
    identity_illumination_phase_ms: int = 850
    identity_illumination_min_frames_per_phase: int = 2
    identity_illumination_min_response: float = 0.008
    identity_illumination_pass_correlation: float = 0.35
    identity_illumination_fail_correlation: float = -0.05

    redis_url: str = "redis://localhost:6379/0"
    queue_name: str = "proctorcore"
    webhook_max_attempts: int = 7
    webhook_retry_intervals: str = "10,30,120,600,1800,7200"

    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "sental-minio"
    s3_secret_key: str = "change-me-minio-secret"
    s3_bucket_temp: str = "proctoring-temp"
    s3_bucket_evidence: str = "proctoring-evidence"
    s3_bucket_reports: str = "proctoring-reports"
    s3_region: str = "us-east-1"
    s3_secure: bool = False

    livekit_url: str = "wss://livekit.example.kz"
    livekit_api_key: str = "dev-livekit-key"
    livekit_api_secret: str = "dev-livekit-secret"
    livekit_internal_url: str = "http://livekit:7880"
    livekit_egress_enabled: bool = False
    livekit_egress_segment_seconds: int = 6
    livekit_egress_health_url: str = "http://livekit-egress:9090/metrics"
    livekit_client_script_url: str = "https://cdn.jsdelivr.net/npm/livekit-client/dist/livekit-client.umd.min.js"

    clip_pre_seconds: int = 15
    clip_post_seconds: int = 15
    media_chunk_retention_seconds: int = 900
    media_upload_token_ttl_seconds: int = 14400
    media_chunk_max_bytes: int = 26214400
    monitor_lookaway_yaw_threshold: float = 0.42
    monitor_request_timeout_seconds: float = 8.0

    default_video_retention_days: int = 30
    default_report_retention_days: int = 183
    default_appeal_period_days: int = 14

    moodle_webhook_url: str = ""
    moodle_webhook_secret: str = ""

    @property
    def production_readiness_required(self) -> bool:
        return bool(self.storage_require_ready or self.app_env.strip().lower() == "production")

    @property
    def cors_origins(self) -> list[str]:
        value = self.cors_origins_raw.strip()
        if value == "*":
            return ["*"]
        return [item.strip() for item in value.split(",") if item.strip()]

    @property
    def webhook_retry_schedule(self) -> list[int]:
        values = []
        for item in self.webhook_retry_intervals.split(","):
            try:
                values.append(max(1, int(item.strip())))
            except ValueError:
                continue
        return values or [10, 30, 120, 600, 1800, 7200]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.local_storage_root.mkdir(parents=True, exist_ok=True)
    settings.identity_model_root.mkdir(parents=True, exist_ok=True)
    return settings
