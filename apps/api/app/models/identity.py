from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class IdentityStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
    NEEDS_RETRY = "needs_retry"


class AccessDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REVIEW = "review"
    RETRY = "retry"


class IdentityVerifyResponse(BaseModel):
    ok: bool = True
    identity_status: IdentityStatus
    access_decision: AccessDecision
    access_allowed: bool
    match_score: float = Field(ge=0.0, le=1.0)
    reason: str
    live_snapshot_key: str
    reference_snapshot_key: str
    quality: dict[str, Any]
    engine: str
