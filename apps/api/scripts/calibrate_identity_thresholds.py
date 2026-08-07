#!/usr/bin/env python3
"""Evaluate identity thresholds from labelled face-image pairs.

CSV format:

    label,image_a,image_b
    1,/path/same_a.jpg,/path/same_b.jpg
    0,/path/person_a.jpg,/path/person_b.jpg

Use webcam-like samples from the client environment. Do not calibrate only on
clean passport/profile photos.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import statistics
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs_csv", type=Path)
    parser.add_argument("--app-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--threshold-step", type=float, default=0.01)
    args = parser.parse_args()

    sys.path.insert(0, str(args.app_dir))
    os.environ.setdefault("APP_ENV", "staging")
    os.environ.setdefault("IDENTITY_ALLOW_LEGACY_MATCHER", "false")

    from app.services.face_matcher import FaceMatcher

    matcher = FaceMatcher()
    same_scores: list[float] = []
    different_scores: list[float] = []
    failures: list[str] = []

    with args.pairs_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=2):
            try:
                label = int(row["label"])
                left = Path(row["image_a"]).read_bytes()
                right = Path(row["image_b"]).read_bytes()
                result = matcher.verify(left, right, pass_threshold=0.0)
                if result.reason not in {"ok", "low_confidence", "mismatch"}:
                    failures.append(f"line {index}: {result.reason}")
                    continue
                if label == 1:
                    same_scores.append(result.score)
                else:
                    different_scores.append(result.score)
            except Exception as exc:  # noqa: BLE001 - CLI should keep processing rows.
                failures.append(f"line {index}: {exc}")

    if not same_scores or not different_scores:
        print("Need at least one valid same-person and one valid different-person pair.", file=sys.stderr)
        return 2

    print(f"engine={matcher.engine}")
    print(f"same_pairs={len(same_scores)} different_pairs={len(different_scores)} failures={len(failures)}")
    print(f"same_score_mean={statistics.mean(same_scores):.4f} min={min(same_scores):.4f}")
    print(f"different_score_mean={statistics.mean(different_scores):.4f} max={max(different_scores):.4f}")
    print()
    print("threshold,false_match_rate,false_non_match_rate")

    threshold = 0.0
    best_review = 0.0
    while threshold <= 1.0001:
        false_matches = sum(1 for score in different_scores if score >= threshold)
        false_non_matches = sum(1 for score in same_scores if score < threshold)
        fmr = false_matches / len(different_scores)
        fnmr = false_non_matches / len(same_scores)
        if fmr <= 0.01:
            best_review = max(best_review, threshold)
        print(f"{threshold:.2f},{fmr:.4f},{fnmr:.4f}")
        threshold += args.threshold_step

    print()
    print(f"suggested_review_threshold_start={best_review:.2f}")
    print("Choose the final pass threshold with the client after reviewing acceptable false-reject/manual-review rates.")
    if failures:
        print()
        print("Failures:")
        for failure in failures[:50]:
            print(f"- {failure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
