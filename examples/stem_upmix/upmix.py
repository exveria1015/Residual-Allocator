# Copyright 2026 Exveria
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


STEM_ORDER = ("vocals", "drums", "bass", "guitar", "piano", "other")
STEM_EXTENSIONS = (".wav", ".flac", ".aiff", ".aif")
CHANNELS_714 = ("FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR", "TFL", "TFR", "TBL", "TBR")
VOL_RE = re.compile(r"(mean|max)_volume:\s*(-?inf|[-+]?\d+(?:\.\d+)?)\s*dB")
PAN_TERM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\*([A-Z]+)")
FRONT_CHANNELS = {"FL", "FR", "FC"}
SPACE_CHANNELS = {"BL", "BR", "SL", "SR", "TFL", "TFR", "TBL", "TBR"}
TOP_CHANNELS = {"TFL", "TFR", "TBL", "TBR"}
STEREO_FOLD_WEIGHTS = {
    "FL": (1.0, 0.0),
    "FR": (0.0, 1.0),
    "FC": (0.7071, 0.7071),
    "BL": (0.50, 0.0),
    "BR": (0.0, 0.50),
    "SL": (0.55, 0.0),
    "SR": (0.0, 0.55),
    "TFL": (0.35, 0.0),
    "TFR": (0.0, 0.35),
    "TBL": (0.25, 0.0),
    "TBR": (0.0, 0.25),
    "LFE": (0.0, 0.0),
}
BAND_NAMES = ("low", "mid", "high")


@dataclass
class VolumeStats:
    mean_db: float
    max_db: float


@dataclass
class StemStats:
    name: str
    path: Path
    full: VolumeStats
    low: VolumeStats
    mid: VolumeStats
    high: VolumeStats
    mono: VolumeStats
    side: VolumeStats

    @property
    def width_db(self) -> float:
        return self.side.mean_db - self.mono.mean_db

    @property
    def low_vs_mid_db(self) -> float:
        return self.low.mean_db - self.mid.mean_db

    @property
    def high_vs_mid_db(self) -> float:
        return self.high.mean_db - self.mid.mean_db


@dataclass
class WindowStats:
    name: str
    active_ratio: float
    p10_db: float
    p50_db: float
    p90_db: float
    sustain_score: float
    envelope_db: tuple[float, ...]


@dataclass
class Placement:
    stem: str
    role: str
    pan: dict[str, str]
    lfe_amount: float = 0.0
    lfe_cutoff_hz: int = 80


def db_to_amp(db: float) -> float:
    if math.isinf(db):
        return 0.0
    return 10.0 ** (db / 20.0)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def fmt(value: float) -> str:
    if abs(value) < 0.0005:
        return "0"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def safe_label(text: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip())
    label = label.strip("_")
    return label or "stem"


def safe_filename(text: str) -> str:
    name = re.sub(r"[^\w .()-]+", "_", text.strip(), flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name.strip("._") or "upmix"


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    kwargs = {
        "text": True,
        "stdout": subprocess.PIPE if capture else None,
        "stderr": subprocess.PIPE if capture else None,
    }
    proc = subprocess.run(cmd, **kwargs)
    if check and proc.returncode != 0:
        if capture:
            if proc.stdout:
                print(proc.stdout, file=sys.stderr)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    return proc


def parse_volume(stderr: str) -> VolumeStats:
    values: dict[str, float] = {}
    for key, raw in VOL_RE.findall(stderr):
        values[key] = float("-inf") if raw == "-inf" else float(raw)
    if "mean" not in values or "max" not in values:
        raise RuntimeError("ffmpeg volumedetect output did not include mean_volume/max_volume")
    return VolumeStats(mean_db=values["mean"], max_db=values["max"])


def volumedetect(ffmpeg: str, path: Path, audio_filter: str) -> VolumeStats:
    proc = run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            audio_filter,
            "-f",
            "null",
            "-",
        ],
        capture=True,
    )
    return parse_volume(proc.stderr or "")


def volumedetect_complex(ffmpeg: str, path: Path, filter_complex: str, label: str) -> VolumeStats:
    proc = run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-filter_complex",
            filter_complex,
            "-map",
            f"[{label}]",
            "-f",
            "null",
            "-",
        ],
        capture=True,
    )
    return parse_volume(proc.stderr or "")


def ffprobe_json(ffprobe: str, path: Path) -> dict:
    proc = run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:format_tags:stream=index,codec_type,codec_name,sample_rate,channels,channel_layout,duration,width,height",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    return json.loads(proc.stdout or "{}")


def media_duration(ffprobe: str, path: Path) -> float:
    data = ffprobe_json(ffprobe, path)
    duration = (data.get("format") or {}).get("duration")
    if duration is not None:
        return float(duration)
    for stream in data.get("streams") or []:
        if stream.get("codec_type") == "audio" and stream.get("duration"):
            return float(stream["duration"])
    raise RuntimeError(f"Could not determine duration for {path}")


def media_tags(ffprobe: str, path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    data = ffprobe_json(ffprobe, path)
    tags = (data.get("format") or {}).get("tags") or {}
    return {str(k).lower(): str(v) for k, v in tags.items()}


def resolve_stem_dir(stem_dir: Path | None) -> Path:
    if stem_dir is None:
        raise SystemExit("Pass --stem-dir.")
    return stem_dir.resolve()


def resolve_input_file(explicit_input: Path | None) -> Path | None:
    return explicit_input.resolve() if explicit_input is not None else None


def find_stems(stem_dir: Path, requested: Iterable[str]) -> dict[str, Path]:
    stems: dict[str, Path] = {}
    for stem in requested:
        for ext in STEM_EXTENSIONS:
            candidate = stem_dir / f"{stem}{ext}"
            if candidate.exists():
                stems[stem] = candidate.resolve()
                break
    if not stems:
        raise SystemExit(f"No requested stems found in {stem_dir}")
    return stems


def analyze_stem_ffmpeg(ffmpeg: str, name: str, path: Path) -> StemStats:
    full = volumedetect(ffmpeg, path, "volumedetect")
    low = volumedetect(ffmpeg, path, "lowpass=f=180,volumedetect")
    mid = volumedetect(ffmpeg, path, "highpass=f=180,lowpass=f=2500,volumedetect")
    high = volumedetect(ffmpeg, path, "highpass=f=2500,volumedetect")
    mono = volumedetect_complex(
        ffmpeg,
        path,
        "[0:a]pan=mono|c0=0.5*FL+0.5*FR,volumedetect[mid]",
        "mid",
    )
    side = volumedetect_complex(
        ffmpeg,
        path,
        "[0:a]pan=mono|c0=0.5*FL-0.5*FR,volumedetect[side]",
        "side",
    )
    return StemStats(name=name, path=path, full=full, low=low, mid=mid, high=high, mono=mono, side=side)


def analyze_stem_numpy(name: str, path: Path) -> StemStats:
    import numpy as np
    import soundfile as sf
    from scipy import signal

    audio, sample_rate = sf.read(path, always_2d=True, dtype="float64")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    def stats_from(data: np.ndarray) -> VolumeStats:
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        rms = float(np.sqrt(np.mean(data * data))) if data.size else 0.0
        peak_db = 20.0 * math.log10(max(peak, 1e-12))
        rms_db = 20.0 * math.log10(max(rms, 1e-12))
        return VolumeStats(mean_db=rms_db, max_db=peak_db)

    def filtered_stats(kind: str, freq: float | tuple[float, float]) -> VolumeStats:
        sos = signal.butter(4, freq, btype=kind, fs=sample_rate, output="sos")
        filtered = signal.sosfilt(sos, audio, axis=0)
        return stats_from(filtered)

    mono = 0.5 * (audio[:, 0] + audio[:, 1])
    side = 0.5 * (audio[:, 0] - audio[:, 1])
    return StemStats(
        name=name,
        path=path,
        full=stats_from(audio),
        low=filtered_stats("lowpass", 180.0),
        mid=filtered_stats("bandpass", (180.0, 2500.0)),
        high=filtered_stats("highpass", 2500.0),
        mono=stats_from(mono),
        side=stats_from(side),
    )


def analyze_stem(ffmpeg: str, name: str, path: Path, backend: str) -> tuple[StemStats, str]:
    if backend in {"auto", "numpy"}:
        try:
            return analyze_stem_numpy(name, path), "numpy"
        except Exception as exc:
            if backend == "numpy":
                raise
            print(f"numpy analysis unavailable for {name}: {exc}. Falling back to ffmpeg.", file=sys.stderr)
    return analyze_stem_ffmpeg(ffmpeg, name, path), "ffmpeg"


def fallback_window_stats(stats: StemStats) -> WindowStats:
    return WindowStats(
        name=stats.name,
        active_ratio=0.5,
        p10_db=stats.full.mean_db - 12.0,
        p50_db=stats.full.mean_db - 3.0,
        p90_db=stats.full.mean_db + 3.0,
        sustain_score=0.5,
        envelope_db=(),
    )


def analyze_window_stats_numpy(
    name: str,
    path: Path,
    *,
    window_ms: float,
    hop_ms: float,
) -> WindowStats:
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(path, always_2d=True, dtype="float64")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    window = max(1, int(sample_rate * window_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    if len(audio) <= window:
        frames = [audio]
    else:
        frames = [audio[start : start + window] for start in range(0, len(audio) - window + 1, hop)]
    rms_values = []
    for frame in frames:
        rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
        rms_values.append(20.0 * math.log10(max(rms, 1e-12)))
    envelope = np.asarray(rms_values, dtype=np.float64)
    p10, p50, p90 = [float(np.percentile(envelope, value)) for value in (10, 50, 90)]
    active_threshold = max(p90 - 24.0, p50 - 9.0, -60.0)
    active_ratio = float(np.mean(envelope >= active_threshold))
    sustain_score = clamp(1.0 - ((p90 - p50) / 18.0), 0.0, 1.0)
    return WindowStats(
        name=name,
        active_ratio=active_ratio,
        p10_db=p10,
        p50_db=p50,
        p90_db=p90,
        sustain_score=sustain_score,
        envelope_db=tuple(float(value) for value in envelope),
    )


def analyze_window_stats(
    name: str,
    path: Path,
    stem_stats: StemStats,
    *,
    window_ms: float,
    hop_ms: float,
) -> WindowStats:
    try:
        return analyze_window_stats_numpy(name, path, window_ms=window_ms, hop_ms=hop_ms)
    except Exception as exc:
        print(f"window analysis unavailable for {name}: {exc}. Using summary stats.", file=sys.stderr)
        return fallback_window_stats(stem_stats)


def envelope_correlation(left: WindowStats | None, right: WindowStats | None) -> float:
    if left is None or right is None or not left.envelope_db or not right.envelope_db:
        return 0.0
    count = min(len(left.envelope_db), len(right.envelope_db))
    if count < 3:
        return 0.0
    import numpy as np

    left_values = np.asarray(left.envelope_db[:count], dtype=np.float64)
    right_values = np.asarray(right.envelope_db[:count], dtype=np.float64)
    left_values -= float(np.mean(left_values))
    right_values -= float(np.mean(right_values))
    denom = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    if denom <= 1e-12:
        return 0.0
    return clamp(float(np.dot(left_values, right_values) / denom), -1.0, 1.0)


def band_overlap_score(left: StemStats, right: StemStats) -> float:
    left_bands = band_fractions(left)
    right_bands = band_fractions(right)
    return clamp(sum(min(left_value, right_value) for left_value, right_value in zip(left_bands, right_bands)), 0.0, 1.0)


def activity_overlap_score(left: WindowStats | None, right: WindowStats | None) -> float:
    if left is None or right is None:
        return 0.5
    return math.sqrt(max(left.active_ratio, 0.0) * max(right.active_ratio, 0.0))


def placement_collision_score(
    left: StemStats | None,
    right: StemStats | None,
    left_window: WindowStats | None,
    right_window: WindowStats | None,
) -> float:
    if left is None or right is None:
        return 0.0
    activity = activity_overlap_score(left_window, right_window)
    temporal = activity * (0.75 * max(0.0, envelope_correlation(left_window, right_window)) + 0.25 * activity)
    spectral = band_overlap_score(left, right)
    width = math.sqrt(wide_score(left) * wide_score(right))
    spread = 0.35 + 0.65 * width
    return clamp(temporal * (0.60 * spectral + 0.40 * spread), 0.0, 1.0)


def vocal_leak_risk(stats: StemStats, vocals: StemStats | None, windows: WindowStats | None, vocal_windows: WindowStats | None) -> float:
    if vocals is None:
        return 0.0
    _low_frac, mid_frac, _high_frac = band_fractions(stats)
    narrow_mid = clamp(0.65 * mid_frac + 0.35 * (1.0 - wide_score(stats)), 0.0, 1.0)
    return placement_collision_score(stats, vocals, windows, vocal_windows) * narrow_mid


def pan_expr(values: dict[str, str]) -> str:
    parts = [f"{channel}={values.get(channel, '0')}" for channel in CHANNELS_714]
    return "pan=7.1.4|" + "|".join(parts)


def wide_score(stats: StemStats) -> float:
    return clamp((stats.width_db + 14.0) / 14.0, 0.0, 1.0)


def bright_score(stats: StemStats) -> float:
    return clamp((stats.high_vs_mid_db + 13.0) / 10.0, 0.0, 1.0)


def low_light_score(stats: StemStats) -> float:
    return clamp((2.0 - stats.low_vs_mid_db) / 16.0, 0.0, 1.0)


def front51_lfe_amounts(lfe_mode: str) -> tuple[float, float]:
    if lfe_mode == "off":
        return 0.0, 0.0
    if lfe_mode == "normal":
        return 0.16, 0.07
    return 0.08, 0.035


def decide_placement_front51(
    stats: StemStats,
    lfe_mode: str,
    window_stats: dict[str, WindowStats],
    all_stats: dict[str, StemStats],
) -> Placement:
    name = stats.name
    wide = wide_score(stats)
    bright = bright_score(stats)
    windows = window_stats.get(name)
    sustain = windows.sustain_score if windows else 0.5
    activity = windows.active_ratio if windows else 0.5
    bass_lfe, drum_lfe = front51_lfe_amounts(lfe_mode)

    if name == "vocals":
        center = 0.20 if stats.width_db <= -10.0 else 0.17
        return Placement(
            stem=name,
            role="front51 vocal anchor",
            pan={
                "FL": "0.50*FL",
                "FR": "0.50*FR",
                "FC": f"{fmt(center)}*FL+{fmt(center)}*FR",
            },
        )

    if name == "drums":
        bass = all_stats.get("bass")
        bass_windows = window_stats.get("bass")
        bass_collision = placement_collision_score(stats, bass, windows, bass_windows)
        side = 0.055 + 0.035 * wide
        side += 0.020 * bass_collision
        center = clamp(0.060 - 0.025 * bass_collision, 0.035, 0.060)
        top = 0.025 + 0.025 * bright
        return Placement(
            stem=name,
            role=f"front51 front rhythm, light room col_bass={bass_collision:.2f}",
            pan={
                "FL": "0.70*FL",
                "FR": "0.70*FR",
                "FC": f"{fmt(center)}*FL+{fmt(center)}*FR",
                "BL": "0.025*FL",
                "BR": "0.025*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TFL": f"{fmt(top)}*FL",
                "TFR": f"{fmt(top)}*FR",
            },
            lfe_amount=drum_lfe * (1.0 - 0.25 * bass_collision),
            lfe_cutoff_hz=95 if lfe_mode == "light" else 115,
        )

    if name == "bass":
        return Placement(
            stem=name,
            role="front51 low anchor, additive LFE",
            pan={
                "FL": "0.18*FL",
                "FR": "0.18*FR",
                "FC": "0.25*FL+0.25*FR",
            },
            lfe_amount=bass_lfe,
            lfe_cutoff_hz=85 if lfe_mode == "light" else 100,
        )

    if name == "piano":
        guitar = all_stats.get("guitar")
        other = all_stats.get("other")
        guitar_collision = placement_collision_score(stats, guitar, windows, window_stats.get("guitar"))
        other_collision = placement_collision_score(stats, other, windows, window_stats.get("other"))
        collision = max(guitar_collision, other_collision)
        side = clamp(0.035 + 0.035 * wide - 0.020 * collision, 0.025, 0.070)
        top = 0.025 + 0.035 * bright
        return Placement(
            stem=name,
            role=f"front51 front-wide piano col={collision:.2f}",
            pan={
                "FL": "0.48*FL",
                "FR": "0.48*FR",
                "FC": "0.015*FL+0.015*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TFL": f"{fmt(top)}*FL",
                "TFR": f"{fmt(top)}*FR",
            },
        )

    if name == "guitar":
        other = all_stats.get("other")
        other_windows = window_stats.get("other")
        other_collision = placement_collision_score(stats, other, windows, other_windows)
        piano = all_stats.get("piano")
        piano_collision = placement_collision_score(stats, piano, windows, window_stats.get("piano"))
        side = clamp(0.14 + 0.12 * wide - 0.045 * other_collision + 0.020 * piano_collision, 0.12, 0.26)
        rear = clamp(0.025 + 0.055 * wide - 0.050 * other_collision, 0.015, 0.08)
        top = 0.015 + 0.025 * bright
        return Placement(
            stem=name,
            role=f"front51 side-front guitar col_other={other_collision:.2f}",
            pan={
                "FL": "0.52*FL",
                "FR": "0.52*FR",
                "BL": f"{fmt(rear)}*FL",
                "BR": f"{fmt(rear)}*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TFL": f"{fmt(top)}*FL",
                "TFR": f"{fmt(top)}*FR",
            },
        )

    if name == "other":
        guitar = all_stats.get("guitar")
        vocals = all_stats.get("vocals")
        piano = all_stats.get("piano")
        guitar_collision = placement_collision_score(stats, guitar, windows, window_stats.get("guitar"))
        piano_collision = placement_collision_score(stats, piano, windows, window_stats.get("piano"))
        vocal_risk = vocal_leak_risk(stats, vocals, windows, window_stats.get("vocals"))
        synth_score = clamp(
            0.34 * wide
            + 0.22 * activity
            + 0.18 * sustain
            + 0.16 * bright
            + 0.10 * low_light_score(stats),
            0.0,
            1.0,
        )
        rear_bias = clamp(synth_score + 0.30 * guitar_collision + 0.12 * piano_collision - 0.25 * vocal_risk, 0.0, 1.0)
        front = clamp(0.26 - 0.08 * rear_bias, 0.16, 0.26)
        side = clamp(0.12 + 0.20 * rear_bias, 0.12, 0.34)
        rear = clamp(0.055 + 0.18 * rear_bias, 0.055, 0.24)
        top_rear = clamp(0.025 + 0.055 * rear_bias, 0.025, 0.08)
        return Placement(
            stem=name,
            role=f"front51 variable synth bed rear={rear_bias:.2f} col_g={guitar_collision:.2f} vocal_risk={vocal_risk:.2f}",
            pan={
                "FL": f"{fmt(front)}*FL",
                "FR": f"{fmt(front)}*FR",
                "BL": f"{fmt(rear)}*FL",
                "BR": f"{fmt(rear)}*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TBL": f"{fmt(top_rear)}*FL",
                "TBR": f"{fmt(top_rear)}*FR",
            },
        )

    front = 0.46
    side = 0.08 + 0.10 * wide
    top = 0.015 + 0.035 * bright
    return Placement(
        stem=name,
        role="front51 generic front bed",
        pan={
            "FL": f"{fmt(front)}*FL",
            "FR": f"{fmt(front)}*FR",
            "SL": f"{fmt(side)}*FL",
            "SR": f"{fmt(side)}*FR",
            "TFL": f"{fmt(top)}*FL",
            "TFR": f"{fmt(top)}*FR",
        },
    )


def decide_placement(
    stats: StemStats,
    lfe_mode: str,
    mix_profile: str,
    window_stats: dict[str, WindowStats],
    all_stats: dict[str, StemStats],
) -> Placement:
    if mix_profile == "front51":
        return decide_placement_front51(stats, lfe_mode, window_stats, all_stats)

    name = stats.name
    wide = stats.width_db > -7.0
    very_wide = stats.width_db > -3.0
    bright = stats.high_vs_mid_db > -8.0

    if lfe_mode == "off":
        bass_lfe = 0.0
        drum_lfe = 0.0
    elif lfe_mode == "normal":
        bass_lfe = 0.18
        drum_lfe = 0.08
    else:
        bass_lfe = 0.045
        drum_lfe = 0.025

    if name == "vocals":
        center = 0.10 if stats.width_db > -10 else 0.13
        return Placement(
            stem=name,
            role="front vocal anchor",
            pan={
                "FL": "0.46*FL",
                "FR": "0.46*FR",
                "FC": f"{fmt(center)}*FL+{fmt(center)}*FR",
                "SL": "0.03*FL",
                "SR": "0.03*FR",
            },
        )

    if name == "drums":
        windows = window_stats.get(name)
        bass = all_stats.get("bass")
        bass_collision = placement_collision_score(stats, bass, windows, window_stats.get("bass"))
        side = 0.30 if wide else 0.26
        side = clamp(side + 0.025 * bass_collision, 0.24, 0.34)
        center = clamp(0.03 - 0.012 * bass_collision, 0.015, 0.03)
        top = 0.10 if bright else 0.08
        return Placement(
            stem=name,
            role=f"front rhythm with side/top energy col_bass={bass_collision:.2f}",
            pan={
                "FL": "0.62*FL",
                "FR": "0.62*FR",
                "FC": f"{fmt(center)}*FL+{fmt(center)}*FR",
                "BL": "0.08*FL",
                "BR": "0.08*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TFL": f"{fmt(top)}*FL",
                "TFR": f"{fmt(top)}*FR",
                "TBL": "0.04*FL",
                "TBR": "0.04*FR",
            },
            lfe_amount=drum_lfe * (1.0 - 0.20 * bass_collision),
            lfe_cutoff_hz=95 if lfe_mode == "light" else 115,
        )

    if name == "bass":
        return Placement(
            stem=name,
            role="front low anchor, LFE as additive support",
            pan={
                "FL": "0.13*FL",
                "FR": "0.13*FR",
                "FC": "0.20*FL+0.20*FR",
            },
            lfe_amount=bass_lfe,
            lfe_cutoff_hz=80 if lfe_mode == "light" else 95,
        )

    if name == "guitar":
        windows = window_stats.get(name)
        other_collision = placement_collision_score(stats, all_stats.get("other"), windows, window_stats.get("other"))
        piano_collision = placement_collision_score(stats, all_stats.get("piano"), windows, window_stats.get("piano"))
        front = clamp(0.46 + 0.035 * other_collision, 0.46, 0.50)
        side = 0.38 if very_wide else (0.34 if wide else 0.26)
        side = clamp(side - 0.045 * other_collision + 0.018 * piano_collision, 0.22, 0.40)
        rear = 0.12 if wide else 0.08
        rear = clamp(rear - 0.055 * other_collision, 0.035, 0.12)
        top = 0.07 if bright else 0.04
        top = clamp(top - 0.020 * other_collision, 0.025, 0.07)
        top_rear = clamp(0.03 - 0.018 * other_collision, 0.012, 0.03)
        return Placement(
            stem=name,
            role=f"wide side instrument col_other={other_collision:.2f}",
            pan={
                "FL": f"{fmt(front)}*FL",
                "FR": f"{fmt(front)}*FR",
                "BL": f"{fmt(rear)}*FL",
                "BR": f"{fmt(rear)}*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TFL": f"{fmt(top)}*FL",
                "TFR": f"{fmt(top)}*FR",
                "TBL": f"{fmt(top_rear)}*FL",
                "TBR": f"{fmt(top_rear)}*FR",
            },
        )

    if name == "piano":
        windows = window_stats.get(name)
        guitar_collision = placement_collision_score(stats, all_stats.get("guitar"), windows, window_stats.get("guitar"))
        other_collision = placement_collision_score(stats, all_stats.get("other"), windows, window_stats.get("other"))
        collision = max(guitar_collision, other_collision)
        front = clamp(0.34 + 0.035 * collision, 0.34, 0.38)
        rear = clamp(0.06 - 0.035 * collision, 0.025, 0.06)
        side = clamp(0.12 - 0.050 * collision, 0.055, 0.12)
        height = 0.18 if bright else 0.14
        height = clamp(height - 0.045 * collision, 0.08, 0.18)
        top_rear = clamp(0.09 - 0.045 * collision, 0.035, 0.09)
        return Placement(
            stem=name,
            role=f"height/front-wide accent col={collision:.2f}",
            pan={
                "FL": f"{fmt(front)}*FL",
                "FR": f"{fmt(front)}*FR",
                "FC": "0.01*FL+0.01*FR",
                "BL": f"{fmt(rear)}*FL",
                "BR": f"{fmt(rear)}*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TFL": f"{fmt(height)}*FL",
                "TFR": f"{fmt(height)}*FR",
                "TBL": f"{fmt(top_rear)}*FL",
                "TBR": f"{fmt(top_rear)}*FR",
            },
        )

    if name == "other":
        windows = window_stats.get(name)
        guitar_collision = placement_collision_score(stats, all_stats.get("guitar"), windows, window_stats.get("guitar"))
        piano_collision = placement_collision_score(stats, all_stats.get("piano"), windows, window_stats.get("piano"))
        vocal_risk = vocal_leak_risk(stats, all_stats.get("vocals"), windows, window_stats.get("vocals"))
        collision = max(guitar_collision, piano_collision)
        rear = 0.26 if wide else 0.22
        rear = clamp(rear + 0.070 * collision - 0.055 * vocal_risk, 0.16, 0.34)
        side = clamp(0.32 + 0.055 * collision - 0.040 * vocal_risk, 0.24, 0.40)
        front = clamp(0.16 - 0.025 * collision + 0.045 * vocal_risk, 0.12, 0.22)
        top_front = clamp(0.07 - 0.025 * vocal_risk, 0.035, 0.07)
        top_rear = clamp(0.20 + 0.055 * collision - 0.040 * vocal_risk, 0.12, 0.26)
        return Placement(
            stem=name,
            role=f"surround bed col={collision:.2f} vocal_risk={vocal_risk:.2f}",
            pan={
                "FL": f"{fmt(front)}*FL",
                "FR": f"{fmt(front)}*FR",
                "BL": f"{fmt(rear)}*FL",
                "BR": f"{fmt(rear)}*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TFL": f"{fmt(top_front)}*FL",
                "TFR": f"{fmt(top_front)}*FR",
                "TBL": f"{fmt(top_rear)}*FL",
                "TBR": f"{fmt(top_rear)}*FR",
            },
        )

    front = 0.42
    side = 0.24 if wide else 0.12
    top = 0.08 if bright else 0.02
    return Placement(
        stem=name,
        role="generic musical bed",
        pan={
            "FL": f"{fmt(front)}*FL",
            "FR": f"{fmt(front)}*FR",
            "SL": f"{fmt(side)}*FL",
            "SR": f"{fmt(side)}*FR",
            "TFL": f"{fmt(top)}*FL",
            "TFR": f"{fmt(top)}*FR",
        },
    )


def gain_filter(gain_db: float) -> str:
    gain = db_to_amp(gain_db)
    if abs(gain - 1.0) < 1e-6:
        return ""
    return f",volume={fmt(gain)}"


def gain_suffix(stem: str, gain_db: float) -> str:
    if abs(gain_db) < 0.05:
        return ""
    direction = "plus" if gain_db > 0 else "minus"
    value = str(round(abs(gain_db), 2)).replace(".", "p").rstrip("0").rstrip("p")
    return f"_{stem}_{direction}{value}db"


def linear_energy(mean_db: float) -> float:
    amp = db_to_amp(mean_db)
    return amp * amp


def band_fractions(stats: StemStats) -> tuple[float, float, float]:
    low = linear_energy(stats.low.mean_db)
    mid = linear_energy(stats.mid.mean_db)
    high = linear_energy(stats.high.mean_db)
    total = max(low + mid + high, 1e-24)
    return low / total, mid / total, high / total


def stats_band_energies(stats: StemStats) -> dict[str, float]:
    return {
        "low": linear_energy(stats.low.mean_db),
        "mid": linear_energy(stats.mid.mean_db),
        "high": linear_energy(stats.high.mean_db),
    }


def db_ratio(numerator: float, denominator: float) -> float:
    return 10.0 * math.log10(max(numerator, 1e-24) / max(denominator, 1e-24))


def tonal_ratios(energies: dict[str, float]) -> tuple[float, float]:
    mid = energies["mid"]
    return db_ratio(energies["low"], mid), db_ratio(energies["high"], mid)


def placement_metrics(placement: Placement) -> tuple[float, float, float]:
    total_sq = 0.0
    space_sq = 0.0
    top_sq = 0.0
    for output_channel, expr in placement.pan.items():
        for raw_coeff, _input_channel in PAN_TERM_RE.findall(expr):
            coeff = float(raw_coeff)
            coeff_sq = coeff * coeff
            total_sq += coeff_sq
            if output_channel in SPACE_CHANNELS:
                space_sq += coeff_sq
            if output_channel in TOP_CHANNELS:
                top_sq += coeff_sq
    if total_sq <= 0.0:
        return 0.0, 0.0, 0.0
    return math.sqrt(total_sq / 2.0), math.sqrt(space_sq / total_sq), math.sqrt(top_sq / total_sq)


def placement_stereo_fold_power(placement: Placement) -> float:
    folded = {
        "FL": [0.0, 0.0],
        "FR": [0.0, 0.0],
    }
    for output_channel, expr in placement.pan.items():
        fold_left, fold_right = STEREO_FOLD_WEIGHTS.get(output_channel, (0.0, 0.0))
        if fold_left == 0.0 and fold_right == 0.0:
            continue
        for raw_coeff, input_channel in PAN_TERM_RE.findall(expr):
            if input_channel not in folded:
                continue
            coeff = float(raw_coeff)
            folded[input_channel][0] += coeff * fold_left
            folded[input_channel][1] += coeff * fold_right

    power = 0.0
    for left_coeff, right_coeff in folded.values():
        power += left_coeff * left_coeff + right_coeff * right_coeff
    return max(power / len(folded), 1e-9)


def spatial_gain_adjustment(placement: Placement) -> float:
    total_power, space_ratio, top_ratio = placement_metrics(placement)
    gain = 0.0
    gain -= 0.45 * max(0.0, space_ratio - 0.40)
    gain -= 0.35 * max(0.0, top_ratio - 0.25)
    gain -= 0.50 * max(0.0, total_power - 0.72)
    return gain


def harman_like_tonal_adjustment(stats: StemStats) -> float:
    low_frac, _mid_frac, high_frac = band_fractions(stats)
    # This is intentionally a small, music-safe tilt. It is not an exact
    # reproduction of any playback target; it just favors low-band anchors and
    # avoids over-prominent bright/wide beds after upmix placement.
    if stats.name == "vocals":
        return clamp(0.10 * low_frac - 0.25 * high_frac, -0.20, 0.10)
    if stats.name == "bass":
        return clamp(0.65 * low_frac - 0.15 * high_frac, 0.0, 0.55)
    if stats.name == "drums":
        return clamp(0.35 * low_frac - 0.15 * high_frac, -0.15, 0.35)
    if stats.name == "guitar":
        return clamp(0.15 * low_frac - 0.55 * high_frac, -0.45, 0.15)
    if stats.name == "piano":
        return clamp(0.10 * low_frac - 0.45 * high_frac, -0.35, 0.10)
    if stats.name == "other":
        return clamp(0.15 * low_frac - 0.45 * high_frac, -0.35, 0.15)
    return clamp(0.20 * low_frac - 0.40 * high_frac, -0.35, 0.20)


def compute_stem_gains(
    stats: list[StemStats],
    placements: dict[str, Placement],
    *,
    mode: str,
    vocal_gain_db: float,
    limit_db: float,
) -> dict[str, float]:
    gains: dict[str, float] = {}
    for item in stats:
        auto_gain = 0.0
        if mode in {"spatial", "harman"}:
            auto_gain += spatial_gain_adjustment(placements[item.name])
        if mode == "harman":
            auto_gain += harman_like_tonal_adjustment(item)
        auto_gain = clamp(auto_gain, -abs(limit_db), abs(limit_db))
        gains[item.name] = auto_gain + (vocal_gain_db if item.name == "vocals" else 0.0)
    return gains


def predicted_fold_band_energies(
    stats: list[StemStats],
    placements: dict[str, Placement],
    stem_gain_db: dict[str, float],
) -> dict[str, float]:
    totals = {band: 0.0 for band in BAND_NAMES}
    for item in stats:
        power = placement_stereo_fold_power(placements[item.name])
        gain_power = db_to_amp(stem_gain_db.get(item.name, 0.0)) ** 2
        energies = stats_band_energies(item)
        for band in BAND_NAMES:
            totals[band] += energies[band] * power * gain_power
    return totals


def match_reference_gains(
    stats: list[StemStats],
    placements: dict[str, Placement],
    reference_stats: StemStats,
    stem_gain_db: dict[str, float],
    limit_db: float,
) -> tuple[dict[str, float], dict[str, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    names = [item.name for item in stats]
    reference_ratios = tonal_ratios(stats_band_energies(reference_stats))
    before_ratios = tonal_ratios(predicted_fold_band_energies(stats, placements, stem_gain_db))
    adjustment_limits = {
        name: min(abs(limit_db), 0.40) if name == "vocals" else abs(limit_db)
        for name in names
    }
    adjustments = {name: 0.0 for name in names}

    def candidate_gains(candidate_adjustments: dict[str, float]) -> dict[str, float]:
        return {
            name: stem_gain_db.get(name, 0.0) + candidate_adjustments.get(name, 0.0)
            for name in names
        }

    def objective(candidate_adjustments: dict[str, float]) -> float:
        ratios = tonal_ratios(predicted_fold_band_energies(stats, placements, candidate_gains(candidate_adjustments)))
        low_error = ratios[0] - reference_ratios[0]
        high_error = ratios[1] - reference_ratios[1]
        regularization = 0.0
        for name, value in candidate_adjustments.items():
            limit = max(adjustment_limits[name], 1e-6)
            weight = 0.45 if name == "vocals" else 0.18
            regularization += weight * (value / limit) ** 2
        return low_error * low_error + 0.85 * high_error * high_error + regularization

    best_score = objective(adjustments)
    for step in (0.25, 0.10, 0.05):
        improved = True
        while improved:
            improved = False
            for name in names:
                limit = adjustment_limits[name]
                best_value = adjustments[name]
                for direction in (-1.0, 1.0):
                    trial = dict(adjustments)
                    trial[name] = clamp(trial[name] + direction * step, -limit, limit)
                    if abs(trial[name] - adjustments[name]) < 1e-9:
                        continue
                    score = objective(trial)
                    if score + 1e-8 < best_score:
                        best_score = score
                        best_value = trial[name]
                if abs(best_value - adjustments[name]) > 1e-9:
                    adjustments[name] = best_value
                    improved = True

    matched = candidate_gains(adjustments)
    after_ratios = tonal_ratios(predicted_fold_band_energies(stats, placements, matched))
    return matched, adjustments, reference_ratios, before_ratios, after_ratios


def print_stem_gains(gains: dict[str, float]) -> None:
    print("Applied stem gains:", flush=True)
    for stem in sorted(gains):
        print(f"{stem:<9} {gains[stem]:>+5.2f} dB", flush=True)


def print_reference_match(
    adjustments: dict[str, float],
    reference_ratios: tuple[float, float],
    before_ratios: tuple[float, float],
    after_ratios: tuple[float, float],
) -> None:
    print("Reference match: original", flush=True)
    print("tonal        low-mid  high-mid", flush=True)
    print(f"original     {reference_ratios[0]:>+7.2f}  {reference_ratios[1]:>+8.2f} dB", flush=True)
    print(f"before       {before_ratios[0]:>+7.2f}  {before_ratios[1]:>+8.2f} dB", flush=True)
    print(f"after        {after_ratios[0]:>+7.2f}  {after_ratios[1]:>+8.2f} dB", flush=True)
    print("Reference stem adjustments:", flush=True)
    for stem in sorted(adjustments):
        print(f"{stem:<9} {adjustments[stem]:>+5.2f} dB", flush=True)


def build_714_filter(
    input_order: list[str],
    placements: dict[str, Placement],
    master_gain: float,
    stem_gain_db: dict[str, float],
) -> str:
    parts: list[str] = []
    mix_labels: list[str] = []
    for index, stem in enumerate(input_order):
        placement = placements[stem]
        gain = gain_filter(stem_gain_db.get(stem, 0.0))
        label = safe_label(stem)
        stem_label = f"{label}_bed"
        if placement.lfe_amount > 0:
            full_label = f"{label}_full"
            low_label = f"{label}_low"
            lfe_label = f"{label}_lfe"
            parts.append(f"[{index}:a]aresample=48000{gain},asplit=2[{full_label}][{low_label}]")
            parts.append(f"[{full_label}]{pan_expr(placement.pan)}[{stem_label}]")
            lfe_pan = {
                "LFE": f"{fmt(placement.lfe_amount)}*FL+{fmt(placement.lfe_amount)}*FR",
            }
            parts.append(f"[{low_label}]lowpass=f={placement.lfe_cutoff_hz},{pan_expr(lfe_pan)}[{lfe_label}]")
            mix_labels.extend([f"[{stem_label}]", f"[{lfe_label}]"])
        else:
            parts.append(f"[{index}:a]aresample=48000{gain},{pan_expr(placement.pan)}[{stem_label}]")
            mix_labels.append(f"[{stem_label}]")
    parts.append(
        "".join(mix_labels)
        + f"amix=inputs={len(mix_labels)}:normalize=0,volume={fmt(master_gain)},alimiter=limit=0.98[out]"
    )
    return ";".join(parts)


def render_714(
    ffmpeg: str,
    stems: dict[str, Path],
    placements: dict[str, Placement],
    output_path: Path,
    master_gain: float,
    stem_gain_db: dict[str, float],
) -> None:
    input_order = [stem for stem in STEM_ORDER if stem in stems] + [stem for stem in stems if stem not in STEM_ORDER]
    filter_complex = build_714_filter(input_order, placements, master_gain, stem_gain_db)
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    for stem in input_order:
        cmd.extend(["-i", str(stems[stem])])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:a",
            "pcm_s24le",
            "-channel_layout",
            "7.1.4",
            str(output_path),
        ]
    )
    run(cmd)


def extract_cover(ffmpeg: str, input_file: Path | None, output_path: Path) -> Path | None:
    if input_file is None or not input_file.exists():
        return None
    proc = run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_file),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-update",
            "1",
            str(output_path),
        ],
        check=False,
        capture=True,
    )
    if proc.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        return None
    return output_path


def fold_714_to_51_filter(input_label: str, output_label: str, lfe_fold_gain: float) -> str:
    return (
        f"{input_label}pan=5.1(side)|"
        "FL=0.92*FL+0.24*FC+0.20*BL+0.18*TFL+0.10*TBL|"
        "FR=0.92*FR+0.24*FC+0.20*BR+0.18*TFR+0.10*TBR|"
        "FC=0.78*FC+0.05*FL+0.05*FR|"
        f"LFE={fmt(lfe_fold_gain)}*LFE|"
        "SL=0.78*SL+0.40*BL+0.20*TFL+0.34*TBL|"
        "SR=0.78*SR+0.40*BR+0.20*TFR+0.34*TBR,"
        f"alimiter=limit=0.98{output_label}"
    )


def lfe_fold_gain_for_mode(lfe_mode: str, mix_profile: str) -> float:
    if lfe_mode == "off":
        return 0.0
    if mix_profile == "front51":
        return 0.65 if lfe_mode == "light" else 0.85
    return 0.45 if lfe_mode == "light" else 0.8


def fold_714_to_stereo_filter(input_label: str, output_label: str, lfe_fold_gain: float, *, limiter: bool) -> str:
    lfe_left = f"+{fmt(lfe_fold_gain)}*LFE" if lfe_fold_gain > 0.0 else ""
    lfe_right = f"+{fmt(lfe_fold_gain)}*LFE" if lfe_fold_gain > 0.0 else ""
    filter_text = (
        f"{input_label}pan=stereo|"
        f"FL=0.92*FL+0.7071*FC+0.50*BL+0.55*SL+0.35*TFL+0.25*TBL{lfe_left}|"
        f"FR=0.92*FR+0.7071*FC+0.50*BR+0.55*SR+0.35*TFR+0.25*TBR{lfe_right},"
        "aresample=48000,aformat=channel_layouts=stereo"
    )
    if limiter:
        filter_text += ",alimiter=limit=0.98"
    return f"{filter_text}{output_label}"


def filename_number_token(value: float) -> str:
    if abs(value) < 0.005:
        return "0"
    sign = "m" if value < 0.0 else "p"
    body = f"{abs(value):.2f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{sign}{body}"


def stereo_loudness_suffix(normalize: str, target_i: float, target_tp: float) -> str:
    if normalize == "off":
        return "stereo"
    return f"stereo_lufs_{filename_number_token(target_i)}_tp_{filename_number_token(target_tp)}"


def parse_loudnorm_json(stderr: str) -> dict[str, str]:
    start = stderr.find("{")
    end = stderr.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("ffmpeg loudnorm output did not include JSON measurements")
    data = json.loads(stderr[start : end + 1])
    required = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"ffmpeg loudnorm JSON is missing: {', '.join(missing)}")
    return {key: str(data[key]) for key in required}


def loudnorm_filter(
    input_label: str,
    output_label: str,
    *,
    target_i: float,
    target_tp: float,
    target_lra: float,
    measurements: dict[str, str] | None,
    print_format: str,
) -> str:
    args = [
        f"I={fmt(target_i)}",
        f"TP={fmt(target_tp)}",
        f"LRA={fmt(target_lra)}",
    ]
    if measurements is not None:
        args.extend(
            [
                f"measured_I={measurements['input_i']}",
                f"measured_TP={measurements['input_tp']}",
                f"measured_LRA={measurements['input_lra']}",
                f"measured_thresh={measurements['input_thresh']}",
                f"offset={measurements['target_offset']}",
                "linear=true",
            ]
        )
    args.append(f"print_format={print_format}")
    return f"{input_label}loudnorm={':'.join(args)},aresample=48000,aformat=channel_layouts=stereo{output_label}"


def render_51_flac(
    ffmpeg: str,
    bed_path: Path,
    output_path: Path,
    tags: dict[str, str],
    title: str,
    lfe_fold_gain: float,
    compression_level: int,
) -> None:
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(bed_path),
        "-filter_complex",
        fold_714_to_51_filter("[0:a]", "[a]", lfe_fold_gain),
        "-map",
        "[a]",
        "-c:a",
        "flac",
        "-compression_level",
        str(compression_level),
        "-metadata",
        f"title={title}",
    ]
    for source_key, flac_key in (("artist", "artist"), ("album", "album"), ("date", "date")):
        value = tags.get(source_key)
        if value:
            cmd.extend(["-metadata", f"{flac_key}={value}"])
    cmd.append(str(output_path))
    run(cmd)


def render_stereo_flac(
    ffmpeg: str,
    bed_path: Path,
    output_path: Path,
    tags: dict[str, str],
    title: str,
    lfe_fold_gain: float,
    compression_level: int,
    normalize: str,
    target_i: float,
    target_tp: float,
    target_lra: float,
) -> None:
    if normalize == "loudnorm":
        first_pass_filter = ";".join(
            [
                fold_714_to_stereo_filter("[0:a]", "[fold]", lfe_fold_gain, limiter=False),
                loudnorm_filter(
                    "[fold]",
                    "[norm]",
                    target_i=target_i,
                    target_tp=target_tp,
                    target_lra=target_lra,
                    measurements=None,
                    print_format="json",
                ),
            ]
        )
        first_pass = run(
            [
                ffmpeg,
                "-hide_banner",
                "-nostats",
                "-i",
                str(bed_path),
                "-filter_complex",
                first_pass_filter,
                "-map",
                "[norm]",
                "-f",
                "null",
                "-",
            ],
            capture=True,
        )
        measurements = parse_loudnorm_json(first_pass.stderr or "")
        filter_complex = ";".join(
            [
                fold_714_to_stereo_filter("[0:a]", "[fold]", lfe_fold_gain, limiter=False),
                loudnorm_filter(
                    "[fold]",
                    "[a]",
                    target_i=target_i,
                    target_tp=target_tp,
                    target_lra=target_lra,
                    measurements=measurements,
                    print_format="summary",
                ),
            ]
        )
        comment = f"Stereo fold-down normalized to {fmt(target_i)} LUFS / {fmt(target_tp)} dBTP"
    else:
        filter_complex = fold_714_to_stereo_filter("[0:a]", "[a]", lfe_fold_gain, limiter=True)
        comment = "Stereo fold-down from upmix bed"

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(bed_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[a]",
        "-c:a",
        "flac",
        "-compression_level",
        str(compression_level),
        "-metadata",
        f"title={title}",
        "-metadata",
        f"comment={comment}",
        str(output_path),
    ]
    for source_key, flac_key in (("artist", "artist"), ("album", "album"), ("date", "date")):
        value = tags.get(source_key)
        if value:
            cmd[-1:-1] = ["-metadata", f"{flac_key}={value}"]
    run(cmd)


def render_apple_tv(
    ffmpeg: str,
    bed_path: Path,
    input_file: Path | None,
    output_path: Path,
    duration: float,
    bitrate: str,
    tags: dict[str, str],
    title: str,
    lfe_fold_gain: float,
) -> None:
    cover_path = extract_cover(ffmpeg, input_file, output_path.with_suffix(".cover.jpg"))
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if cover_path is not None:
        cmd.extend(["-f", "image2", "-loop", "1", "-framerate", "1", "-i", str(cover_path)])
        video_filter = (
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v]"
        )
    else:
        cmd.extend(["-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=1"])
        video_filter = "[0:v]format=yuv420p[v]"
    cmd.extend(["-i", str(bed_path)])
    audio_filter = fold_714_to_51_filter("[1:a]", "[a]", lfe_fold_gain)
    cmd.extend(
        [
            "-filter_complex",
            f"{video_filter};{audio_filter}",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-level",
            "4.1",
            "-preset",
            "medium",
            "-tune",
            "stillimage",
            "-crf",
            "20",
            "-r",
            "24",
            "-c:a",
            "eac3",
            "-b:a",
            bitrate,
            "-tag:a",
            "ec-3",
            "-metadata:s:a:0",
            "title=Dolby Digital Plus 5.1 Auto placement",
            "-metadata",
            f"title={title}",
        ]
    )
    for source_key, mp4_key in (("artist", "artist"), ("album", "album"), ("date", "date")):
        value = tags.get(source_key)
        if value:
            cmd.extend(["-metadata", f"{mp4_key}={value}"])
    cmd.extend(["-movflags", "+faststart", "-brand", "mp42", str(output_path)])
    run(cmd)


def print_analysis(stats: list[StemStats], placements: dict[str, Placement]) -> None:
    print("Stem analysis:", flush=True)
    print("stem       rms    low-mid  high-mid  side-mid  role", flush=True)
    for item in stats:
        placement = placements[item.name]
        print(
            f"{item.name:<9} "
            f"{item.full.mean_db:>6.1f} "
            f"{item.low_vs_mid_db:>8.1f} "
            f"{item.high_vs_mid_db:>9.1f} "
            f"{item.width_db:>8.1f}  "
            f"{placement.role}",
            flush=True,
        )


def print_window_analysis(stats: list[StemStats], windows: dict[str, WindowStats]) -> None:
    if not windows:
        return
    print("Window analysis:", flush=True)
    print("stem      active    p50     p90  sustain", flush=True)
    for item in stats:
        window = windows.get(item.name)
        if window is None:
            continue
        print(
            f"{item.name:<9} "
            f"{window.active_ratio:>6.2f} "
            f"{window.p50_db:>7.1f} "
            f"{window.p90_db:>7.1f} "
            f"{window.sustain_score:>7.2f}",
            flush=True,
        )


def print_collision_analysis(stats: list[StemStats], windows: dict[str, WindowStats]) -> None:
    if not windows:
        return
    by_name = {item.name: item for item in stats}
    pairs = [
        ("guitar", "other"),
        ("piano", "guitar"),
        ("piano", "other"),
        ("vocals", "other"),
        ("drums", "bass"),
    ]
    rows = []
    for left_name, right_name in pairs:
        left = by_name.get(left_name)
        right = by_name.get(right_name)
        if left is None or right is None:
            continue
        collision = placement_collision_score(left, right, windows.get(left_name), windows.get(right_name))
        correlation = max(0.0, envelope_correlation(windows.get(left_name), windows.get(right_name)))
        band = band_overlap_score(left, right)
        rows.append((collision, left_name, right_name, correlation, band))
    if not rows:
        return
    print("Placement collision:", flush=True)
    print("pair            score  env    band", flush=True)
    for collision, left_name, right_name, correlation, band in sorted(rows, reverse=True):
        print(
            f"{left_name}+{right_name:<10} {collision:>5.2f}  {correlation:>4.2f}  {band:>5.2f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze separated stems and create a simple music-oriented 7.1.4 / Apple TV upmix."
    )
    parser.add_argument(
        "--stem-dir",
        type=Path,
        help="Directory containing separated stem files such as vocals.wav.",
    )
    parser.add_argument("--input", type=Path, help="Original input audio, used for cover art and metadata.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for upmix outputs. Default: <stem-dir>/upmix.",
    )
    parser.add_argument(
        "--stems",
        default=",".join(STEM_ORDER),
        help="Comma-separated stem names to use. Default: vocals,drums,bass,guitar,piano,other.",
    )
    parser.add_argument(
        "--lfe-mode",
        choices=("light", "normal", "off"),
        default="light",
        help="LFE strategy. light treats LFE as subtle additive support. Default: light.",
    )
    parser.add_argument(
        "--analysis-backend",
        choices=("auto", "numpy", "ffmpeg"),
        default="auto",
        help="Stem analysis backend. auto uses numpy/soundfile/scipy when available, then ffmpeg. Default: auto.",
    )
    parser.add_argument(
        "--mix-profile",
        choices=("auto", "front51"),
        default="auto",
        help="Spatial placement profile. front51 keeps the main image forward and moves synth-like other rearward. Default: auto.",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=1000.0,
        help="RMS window size for analysis-driven static placement. Default: 1000 ms.",
    )
    parser.add_argument(
        "--hop-ms",
        type=float,
        default=500.0,
        help="RMS hop size for analysis-driven static placement. Default: 500 ms.",
    )
    parser.add_argument(
        "--vocal-gain-db",
        type=float,
        default=None,
        help=(
            "Additional gain applied only to vocals before upmix placement. "
            "Default: +1.0 dB, or 0.0 dB with --reference-match original."
        ),
    )
    parser.add_argument(
        "--stem-gain-mode",
        choices=("off", "spatial", "harman"),
        default="off",
        help=(
            "Optional automatic per-stem gain. spatial compensates for wide/height placement; "
            "harman adds a conservative Harman-inspired tonal tilt. Default: off."
        ),
    )
    parser.add_argument(
        "--stem-gain-limit-db",
        type=float,
        default=1.5,
        help="Maximum automatic per-stem gain adjustment before vocal gain is added. Default: 1.5 dB.",
    )
    parser.add_argument(
        "--reference-match",
        choices=("off", "original"),
        default="off",
        help="Match the predicted stereo fold-down tonal balance to a reference. Default: off.",
    )
    parser.add_argument(
        "--reference-match-limit-db",
        type=float,
        default=1.0,
        help="Maximum per-stem gain adjustment used by --reference-match original. Default: 1.0 dB.",
    )
    parser.add_argument("--master-gain", type=float, default=0.84, help="Final 7.1.4 gain before limiting. Default: 0.84.")
    parser.add_argument(
        "--flac-compression-level",
        type=int,
        default=8,
        help="Compression level for the default 5.1 FLAC output. Default: 8.",
    )
    parser.add_argument("--skip-flac", action="store_true", help="Do not render the default 5.1 FLAC output.")
    parser.add_argument("--skip-stereo", action="store_true", help="Do not render the stereo fold-down FLAC output.")
    parser.add_argument(
        "--stereo-normalize",
        choices=("loudnorm", "off"),
        default="loudnorm",
        help="Stereo fold-down normalization. loudnorm uses a two-pass EBU R128 pass. Default: loudnorm.",
    )
    parser.add_argument(
        "--stereo-loudness-i",
        type=float,
        default=-14.0,
        help="Stereo fold-down integrated loudness target in LUFS. Default: -14.0.",
    )
    parser.add_argument(
        "--stereo-loudness-tp",
        type=float,
        default=-1.0,
        help="Stereo fold-down true peak target in dBTP. Positive values are treated as headroom, so 1.0 means -1.0. Default: -1.0.",
    )
    parser.add_argument(
        "--stereo-loudness-lra",
        type=float,
        default=11.0,
        help="Stereo fold-down loudness range target for loudnorm. Default: 11.0.",
    )
    parser.add_argument(
        "--stereo-lfe-fold-gain",
        type=float,
        default=0.0,
        help="How much LFE is folded into the stereo output. Default: 0.0 because LFE is additive support.",
    )
    parser.add_argument("--apple-tv-bitrate", default="768k", help="E-AC-3 bitrate for Apple TV MP4. Default: 768k.")
    parser.add_argument("--skip-apple-tv", action="store_true", help="Do not render the Apple TV E-AC-3 MP4 output.")
    parser.add_argument("--skip-bed", action="store_true", help="Do not keep the intermediate 7.1.4 WAV after derived outputs.")
    parser.add_argument("--title", help="Override output title metadata.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable. Default: ffmpeg.")
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable. Default: ffprobe.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stem_dir = resolve_stem_dir(args.stem_dir)
    input_file = resolve_input_file(args.input)
    output_dir = args.output_dir.resolve() if args.output_dir else stem_dir / "upmix"
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = [stem.strip() for stem in args.stems.split(",") if stem.strip()]
    stems = find_stems(stem_dir, requested)
    ignored = sorted(path.stem for path in stem_dir.iterdir() if path.is_file() and path.stem == "instrumental")
    if ignored:
        print("Ignoring instrumental stem to avoid double-counting derived mix-minus-vocals audio.", flush=True)

    stats: list[StemStats] = []
    backends: set[str] = set()
    for stem in stems:
        item, backend = analyze_stem(args.ffmpeg, stem, stems[stem], args.analysis_backend)
        stats.append(item)
        backends.add(backend)
    print(f"Analysis backend: {', '.join(sorted(backends))}", flush=True)
    stats_by_name = {item.name: item for item in stats}
    window_stats: dict[str, WindowStats] = {}
    if args.mix_profile in {"auto", "front51"}:
        window_stats = {
            item.name: analyze_window_stats(
                item.name,
                stems[item.name],
                item,
                window_ms=float(args.window_ms),
                hop_ms=float(args.hop_ms),
            )
            for item in stats
        }
        print_window_analysis(stats, window_stats)
        print_collision_analysis(stats, window_stats)
    placements = {
        item.name: decide_placement(item, args.lfe_mode, args.mix_profile, window_stats, stats_by_name)
        for item in stats
    }
    print_analysis(stats, placements)

    source_title = args.title
    tags = media_tags(args.ffprobe, input_file)
    if source_title is None:
        source_title = tags.get("title") or stem_dir.name
    slug = safe_filename(source_title)
    if args.vocal_gain_db is None:
        vocal_gain_db = 0.0 if args.reference_match == "original" else 1.0
    else:
        vocal_gain_db = float(args.vocal_gain_db)
    stem_gain_db = compute_stem_gains(
        stats,
        placements,
        mode=args.stem_gain_mode,
        vocal_gain_db=vocal_gain_db,
        limit_db=float(args.stem_gain_limit_db),
    )
    reference_suffix = ""
    if args.reference_match == "original":
        if input_file is None:
            raise SystemExit("--reference-match original requires --input.")
        reference_stats, reference_backend = analyze_stem(args.ffmpeg, "original", input_file, args.analysis_backend)
        print(f"Reference analysis backend: {reference_backend}", flush=True)
        stem_gain_db, reference_adjustments, reference_ratios, before_ratios, after_ratios = match_reference_gains(
            stats,
            placements,
            reference_stats,
            stem_gain_db,
            float(args.reference_match_limit_db),
        )
        print_reference_match(reference_adjustments, reference_ratios, before_ratios, after_ratios)
        reference_suffix = "_ref_original"
    print_stem_gains(stem_gain_db)
    stem_gain_suffix = f"_stemgain_{args.stem_gain_mode}" if args.stem_gain_mode != "off" else ""
    profile_root = "front51" if args.mix_profile == "front51" else "auto_placement"
    profile = (
        f"{profile_root}_lfe_{args.lfe_mode}"
        f"{gain_suffix('vocal', vocal_gain_db)}"
        f"{stem_gain_suffix}{reference_suffix}"
    )
    bed_path = output_dir / f"{slug}_7.1.4_{profile}.wav"
    flac_path = output_dir / f"{slug}_{profile}_upmix5.1.flac"
    stereo_target_i = float(args.stereo_loudness_i)
    stereo_target_tp = -abs(float(args.stereo_loudness_tp))
    stereo_suffix = stereo_loudness_suffix(args.stereo_normalize, stereo_target_i, stereo_target_tp)
    stereo_path = output_dir / f"{slug}_{profile}_{stereo_suffix}.flac"
    apple_path = output_dir / f"{slug}_{profile}_ddp5.1_apple_tv.mp4"

    print(f"Rendering 7.1.4 WAV: {bed_path}", flush=True)
    render_714(args.ffmpeg, stems, placements, bed_path, args.master_gain, stem_gain_db)
    lfe_fold_gain = lfe_fold_gain_for_mode(args.lfe_mode, args.mix_profile)

    if not args.skip_flac:
        print(f"Rendering 5.1 FLAC: {flac_path}", flush=True)
        render_51_flac(
            args.ffmpeg,
            bed_path,
            flac_path,
            tags,
            source_title,
            lfe_fold_gain,
            int(args.flac_compression_level),
        )

    if not args.skip_stereo:
        print(f"Rendering stereo FLAC: {stereo_path}", flush=True)
        render_stereo_flac(
            args.ffmpeg,
            bed_path,
            stereo_path,
            tags,
            source_title,
            max(0.0, float(args.stereo_lfe_fold_gain)),
            int(args.flac_compression_level),
            args.stereo_normalize,
            stereo_target_i,
            stereo_target_tp,
            float(args.stereo_loudness_lra),
        )

    if not args.skip_apple_tv:
        duration = media_duration(args.ffprobe, bed_path)
        print(f"Rendering Apple TV MP4: {apple_path}", flush=True)
        render_apple_tv(
            args.ffmpeg,
            bed_path,
            input_file,
            apple_path,
            duration,
            args.apple_tv_bitrate,
            tags,
            source_title,
            lfe_fold_gain,
        )
    if args.skip_bed and (not args.skip_flac or not args.skip_stereo or not args.skip_apple_tv):
        bed_path.unlink(missing_ok=True)

    print("Done.", flush=True)
    if bed_path.exists():
        print(f"7.1.4: {bed_path}", flush=True)
    if not args.skip_flac:
        print(f"5.1 FLAC: {flac_path}", flush=True)
    if not args.skip_stereo:
        print(f"Stereo FLAC: {stereo_path}", flush=True)
    if not args.skip_apple_tv:
        print(f"Apple TV: {apple_path}", flush=True)


if __name__ == "__main__":
    main()
