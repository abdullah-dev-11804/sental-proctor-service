#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import statistics
import time
from pathlib import Path
from uuid import uuid4

import httpx


async def main() -> None:
    parser = argparse.ArgumentParser(description="Measure real monitoring inference under concurrent sessions.")
    parser.add_argument("frame", type=Path, help="Consented JPEG/PNG face frame used for the probe.")
    parser.add_argument("--url", default="https://proctoring.sental.kz")
    parser.add_argument("--token", default=os.getenv("API_SHARED_SECRET", ""))
    parser.add_argument("--company-id", type=int, required=True)
    parser.add_argument("--concurrency", type=int, choices=[5, 10, 20], required=True)
    parser.add_argument("--samples-per-session", type=int, default=10)
    args = parser.parse_args()
    if not args.token:
        parser.error("Provide --token or API_SHARED_SECRET.")
    image = args.frame.read_bytes()
    if len(image) < 256:
        parser.error("The frame is empty or too small.")

    base = args.url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.token}"}
    encoded = base64.b64encode(image).decode("ascii")
    latencies: list[float] = []
    errors: list[str] = []

    async with httpx.AsyncClient(headers=headers, timeout=20.0) as client:
        sessions = []
        for index in range(args.concurrency):
            response = await client.post(f"{base}/api/v1/sessions", json={
                "companyId": args.company_id,
                "attemptId": 900000 + index,
                "user": {"id": 900000 + index},
                "source": {"type": "load_probe", "runId": uuid4().hex},
            })
            response.raise_for_status()
            sessions.append(response.json()["session"]["sessionId"])

        async def run_session(session_id: str) -> None:
            for _sample in range(max(1, args.samples_per_session)):
                started = time.perf_counter()
                try:
                    response = await client.post(f"{base}/api/v1/monitor/analyse", json={
                        "sessionId": session_id,
                        "companyId": args.company_id,
                        "frameImage": encoded,
                    })
                    response.raise_for_status()
                except Exception as exc:
                    errors.append(str(exc)[:300])
                finally:
                    latencies.append((time.perf_counter() - started) * 1000.0)

        run_started = time.perf_counter()
        await asyncio.gather(*(run_session(session_id) for session_id in sessions))
        elapsed = time.perf_counter() - run_started

    ordered = sorted(latencies)
    percentile = lambda fraction: ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]
    print(json.dumps({
        "concurrency": args.concurrency,
        "requests": len(latencies),
        "errors": len(errors),
        "errorRate": round(len(errors) / max(1, len(latencies)), 4),
        "elapsedSeconds": round(elapsed, 3),
        "requestsPerSecond": round(len(latencies) / max(0.001, elapsed), 3),
        "latencyMs": {
            "mean": round(statistics.mean(latencies), 2),
            "p50": round(percentile(0.50), 2),
            "p95": round(percentile(0.95), 2),
            "max": round(max(latencies), 2),
        },
        "sampleErrors": errors[:5],
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
