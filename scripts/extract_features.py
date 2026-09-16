#!/usr/bin/env python3
"""Extract transposition-aware harmonic features from the validated WAV data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_domains import CLASSES, summarize  # noqa: E402


FEATURE_VERSION = "harmonic_rf_baseline_v1"
FEATURE_CONFIG = {
    "version": FEATURE_VERSION,
    "harmonic_features": [
        "mean chroma_cqt, sum-normalized, circularly shifted so dominant pitch class is index 0",
        "CQT pitch-class energy, sum-normalized, circularly shifted by the same dominant pitch class",
        "elementwise product of the two canonical harmonic vectors",
    ],
    "diagnostic_features": [
        "spectral_centroid_hz / 1000",
        "spectral_bandwidth_hz / 1000",
        "spectral_rolloff_hz / 1000",
        "zero_crossing_rate",
    ],
    "excluded_features": [
        "duration_s",
        "sample_rate",
        "rms_mean",
        "rms_p10",
        "rms_p50",
        "rms_p90",
        "low_energy_proportion",
    ],
    "canonicalization": "rotate each 12-bin vector left by argmax(mean chroma_cqt)",
}


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    return values / max(float(values.sum()), 1e-12)


def feature_names() -> list[str]:
    names = [f"chroma_root_norm_{index}" for index in range(12)]
    names += [f"cqt_root_norm_{index}" for index in range(12)]
    names += [f"harmonic_product_{index}" for index in range(12)]
    names += [
        "spectral_centroid_khz",
        "spectral_bandwidth_khz",
        "spectral_rolloff_khz",
        "zero_crossing_rate",
    ]
    return names


def extract_features(path: Path) -> np.ndarray:
    row = summarize(path)
    chroma = normalized(np.asarray([row[f"chroma_{index}"] for index in range(12)]))
    cqt = normalized(np.asarray([row[f"cqt_pitch_class_{index}"] for index in range(12)]))
    root = int(np.argmax(chroma))
    chroma = np.roll(chroma, -root)
    cqt = np.roll(cqt, -root)
    diagnostics = np.asarray(
        [
            float(row["spectral_centroid_hz"]) / 1000.0,
            float(row["spectral_bandwidth_hz"]) / 1000.0,
            float(row["spectral_rolloff_hz"]) / 1000.0,
            float(row["zero_crossing_rate"]),
        ],
        dtype=float,
    )
    return np.concatenate((chroma, cqt, chroma * cqt, diagnostics))


def collect_synthetic(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    for category in CLASSES:
        files = sorted((dataset_root / category).glob("*.wav"))
        if not files:
            raise FileNotFoundError(f"No WAV files found for class {category}: {dataset_root / category}")
        for path in files:
            vectors.append(extract_features(path))
            labels.append(category)
            paths.append(str(path.relative_to(ROOT)))
    return np.asarray(vectors), np.asarray(labels), paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-root", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "synthetic_features.npz")
    parser.add_argument("--config", type=Path, default=ROOT / "model" / "feature_config.json")
    args = parser.parse_args()
    features, labels, paths = collect_synthetic(args.synthetic_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=features, labels=labels, paths=np.asarray(paths))
    args.config.write_text(
        json.dumps({"feature_names": feature_names(), **FEATURE_CONFIG}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Extracted {len(labels)} synthetic examples with {features.shape[1]} features")
    print(f"Wrote {args.output.relative_to(ROOT)} and {args.config.relative_to(ROOT)}")


if __name__ == "__main__":
    main()