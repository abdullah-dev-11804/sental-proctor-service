#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Proctoring Server production readiness.")
    parser.add_argument("--url", default="https://proctoring.sental.kz")
    parser.add_argument("--token", default=os.getenv("API_SHARED_SECRET", ""))
    parser.add_argument("--session-id")
    parser.add_argument("--company-id", type=int)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    response = httpx.get(f"{base}/api/health", timeout=15.0)
    response.raise_for_status()
    body = response.json()
    features = body.get("features") or {}
    failures = []

    required = {
        "service_ready": body.get("ok") is True and body.get("status") == "healthy",
        "identity_models": features.get("identity_models_ready") is True,
        "private_storage": (features.get("storage") or {}).get("ready") is True
        and (features.get("storage") or {}).get("private") is True,
        "redis": (features.get("redis") or {}).get("ready") is True,
        "worker": (features.get("worker") or {}).get("ready") is True
        and int((features.get("worker") or {}).get("workers") or 0) > 0,
        "egress": (features.get("egress") or {}).get("ready") is True,
        "webhook": (features.get("webhook") or {}).get("ready") is True,
    }
    failures.extend(name for name, ready in required.items() if not ready)

    diagnostics = None
    if args.session_id:
        if args.company_id is None or not args.token:
            parser.error("--session-id requires --company-id and --token/API_SHARED_SECRET")
        diagnostics_response = httpx.get(
            f"{base}/api/v1/sessions/{args.session_id}",
            headers={
                "Authorization": f"Bearer {args.token}",
                "X-ProctorCore-Company": str(args.company_id),
            },
            timeout=15.0,
        )
        diagnostics_response.raise_for_status()
        diagnostics = diagnostics_response.json()

    result = {
        "ok": not failures,
        "checks": required,
        "failures": failures,
        "health": body,
        "sessionDiagnostics": diagnostics,
    }
    print(json.dumps(result, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
