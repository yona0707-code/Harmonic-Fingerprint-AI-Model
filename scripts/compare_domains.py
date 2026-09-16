#!/usr/bin/env python3
"""Compare real reference WAVs with synthetic audio without training a model."""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import librosa
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RATE = 44_100
CLASSES = ("major", "minor", "ninth", "diminished")


def summarize(path: Path) -> dict[str, float | str]:
    audio, rate = librosa.load(path, sr=None, mono=True)
    rms = librosa.feature.rms(y=audio, frame_length=4096, hop_length=2048)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=rate, n_fft=4096, hop_length=2048)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=rate, n_fft=4096, hop_length=2048)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=rate, roll_percent=0.85, n_fft=4096, hop_length=2048)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=4096, hop_length=2048)[0]
    chroma = librosa.feature.chroma_cqt(y=audio, sr=rate, hop_length=2048)
    cqt = np.abs(librosa.cqt(y=audio, sr=rate, hop_length=2048))
    pitch_class_energy = np.zeros(12)
    for index in range(cqt.shape[0]):
        pitch_class_energy[index % 12] += float(np.mean(cqt[index]))
    pitch_class_energy /= max(pitch_class_energy.sum(), 1e-12)
    low_energy = float(np.mean(rms < max(np.percentile(rms, 25) * 0.25, 1e-5)))
    result: dict[str, float | str] = {
        "sample_rate": float(rate), "duration_s": len(audio) / rate,
        "rms_mean": float(np.mean(rms)), "rms_p10": float(np.percentile(rms, 10)),
        "rms_p50": float(np.percentile(rms, 50)), "rms_p90": float(np.percentile(rms, 90)),
        "low_energy_proportion": low_energy, "spectral_centroid_hz": float(np.mean(centroid)),
        "spectral_bandwidth_hz": float(np.mean(bandwidth)), "spectral_rolloff_hz": float(np.mean(rolloff)),
        "zero_crossing_rate": float(np.mean(zcr)),
    }
    result.update({f"chroma_{index}": float(np.mean(chroma[index])) for index in range(12)})
    result.update({f"cqt_pitch_class_{index}": float(pitch_class_energy[index]) for index in range(12)})
    return result


def main() -> None:
    files: list[tuple[str, str, Path]] = []
    for category in CLASSES:
        files.append(("real", category, ROOT / "data" / "real_reference" / f"{category}.wav"))
        files.extend(("synthetic", category, path) for path in sorted((ROOT / "data" / "synthetic" / category).glob("*.wav")))
    rows = []
    for domain, category, path in files:
        if path.exists():
            rows.append({"domain": domain, "class": category, "file": str(path.relative_to(ROOT)), **summarize(path)})
    report = ROOT / "reports" / "domain_comparison.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = [key for key in rows[0] if key not in {"domain", "class", "file"}]
    lines = ["# Synthetic vs Real Audio Diagnostics", "", "No classifier or training features were created.", "", "## Summary"]
    for domain, category, group in (("real", c, [r for r in rows if r["domain"] == "real" and r["class"] == c]) for c in CLASSES):
        row = group[0]
        lines.append(f"- Real `{category}`: {row['sample_rate']:.0f} Hz, {row['duration_s']:.6f} s, RMS mean {row['rms_mean']:.5f}, low-energy {row['low_energy_proportion']:.3f}, centroid {row['spectral_centroid_hz']:.1f} Hz")
    for category in (*CLASSES, "overall"):
        group = [r for r in rows if r["domain"] == "synthetic" and (category == "overall" or r["class"] == category)]
        lines.append(f"\n### Synthetic {category}")
        for key in numeric:
            values = np.asarray([float(row[key]) for row in group])
            lines.append(f"- `{key}`: mean {values.mean():.6g}, p10 {np.percentile(values, 10):.6g}, p90 {np.percentile(values, 90):.6g}")
    lines += ["", "The CSV contains the complete per-file measurements, including chroma and CQT pitch-class energy summaries."]
    (ROOT / "reports" / "domain_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report} and reports/domain_comparison.md")


if __name__ == "__main__":
    main()