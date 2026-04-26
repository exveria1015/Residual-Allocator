# Copyright 2026 Exveria
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable

warnings.filterwarnings("ignore")

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from models.bs_roformer import BSRoformer
from models.residual_allocator import ConvResidualAllocator, upgrade_allocator_state_dict

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "bs_roformer_sw_fixed_alloc.yaml"
DEFAULT_BASE_CHECKPOINT = REPO_ROOT / "weights" / "BS-Rofo-SW-Fixed.ckpt"
DEFAULT_ALLOCATOR_CONFIG = REPO_ROOT / "configs" / "residual_allocator.yaml"
DEFAULT_ALLOCATOR_CHECKPOINT = REPO_ROOT / "weights" / "residual_allocator.safetensors"
DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif")
ProgressCallback = Callable[[str, int], None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone BS-Roformer inference with optional residual allocator refinement."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input audio file or directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output root. By default, stems are saved under <output>/<input-stem>/.",
    )
    parser.add_argument(
        "--output-dir-is-stem-dir",
        action="store_true",
        help="Write stems directly into --output instead of creating an <input-stem> subdirectory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --input is a directory, search for audio files recursively.",
    )
    parser.add_argument(
        "--input-extensions",
        default=",".join(ext.lstrip(".") for ext in DEFAULT_AUDIO_EXTENSIONS),
        help=(
            "Comma-separated audio extensions used when --input is a directory. "
            "Default: wav,flac,mp3,m4a,aac,ogg,opus,aiff,aif."
        ),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help=f"Base BS-Roformer YAML config. Default: {DEFAULT_BASE_CONFIG}",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=DEFAULT_BASE_CHECKPOINT,
        help=f"Base BS-Roformer checkpoint. Default: {DEFAULT_BASE_CHECKPOINT}",
    )
    parser.add_argument(
        "--allocator-checkpoint",
        type=Path,
        default=DEFAULT_ALLOCATOR_CHECKPOINT,
        help=f"Residual allocator checkpoint. Default: {DEFAULT_ALLOCATOR_CHECKPOINT}",
    )
    parser.add_argument(
        "--allocator-config",
        type=Path,
        default=DEFAULT_ALLOCATOR_CONFIG,
        help=(
            "Residual allocator sidecar YAML config. "
            f"Default: {DEFAULT_ALLOCATOR_CONFIG}. "
            "When omitted/missing, legacy .ckpt files may still provide run_config internally."
        ),
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Run plain BS-Roformer inference without loading or applying the residual allocator.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Inference device. Default: auto",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index when --device is auto/cuda. Default: 0",
    )
    parser.add_argument(
        "--use-tta",
        action="store_true",
        help="Enable test-time augmentation for the base separator.",
    )
    parser.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="Save residual allocator diagnostics under refiner/.",
    )
    parser.add_argument(
        "--extract-instrumental",
        action="store_true",
        help="Save instrumental = mix - vocals when vocals is available.",
    )
    parser.add_argument(
        "--save-instrumental-variants",
        action="store_true",
        help="Also save instrumental_sum_non_vocals and instrumental_mix_minus_vocals.",
    )
    parser.add_argument(
        "--save-mixture-minus-instrumental-sum-non-vocals",
        action="store_true",
        help="Also save mixture_minus_instrumental_sum_non_vocals.",
    )
    parser.add_argument(
        "--collapse-to-4stem",
        action="store_true",
        help="Additionally save bass/drums/other/vocals collapsed from 6 stems.",
    )
    parser.add_argument(
        "--output-4stem-plus-instrumental",
        action="store_true",
        help="Save only bass/drums/other/vocals/instrumental.",
    )
    parser.add_argument(
        "--skip-silent-stems",
        action="store_true",
        help="Do not save stems whose final waveform peak is below the configured silence threshold.",
    )
    parser.add_argument(
        "--allocator-prune-base-silent-stems",
        action="store_true",
        help="Before allocator refinement, zero base stems whose perceptual RMS is below the configured threshold.",
    )
    parser.add_argument(
        "--allocator-base-silent-threshold-db",
        type=float,
        default=-90.0,
        help="Perceptual RMS threshold in dBFS used by --allocator-prune-base-silent-stems. Default: -90.0",
    )
    parser.add_argument(
        "--allocator-base-silent-window-ms",
        type=float,
        default=80.0,
        help="Sliding RMS window in milliseconds used by --allocator-prune-base-silent-stems. Default: 80.0",
    )
    parser.add_argument(
        "--allocator-residualize-base-low-sections",
        action="store_true",
        help="Before allocator routing, zero local base sections whose short-time RMS is below the configured threshold so they return to the residual pool.",
    )
    parser.add_argument(
        "--allocator-base-low-section-threshold-db",
        type=float,
        default=-44.0,
        help="Short-time RMS threshold in dBFS used by --allocator-residualize-base-low-sections. Default: -44.0",
    )
    parser.add_argument(
        "--allocator-base-low-section-window-ms",
        type=float,
        default=80.0,
        help="Sliding RMS window in milliseconds used by --allocator-residualize-base-low-sections. Default: 80.0",
    )
    parser.add_argument(
        "--allocator-base-low-section-fade-protect-ms",
        type=float,
        default=250.0,
        help="Keep low-level sections near active content within this many milliseconds to avoid cutting fades. Default: 250.0",
    )
    parser.add_argument(
        "--allocator-base-low-section-pre-protect-ms",
        type=float,
        default=None,
        help="Optional pre-roll protect width in milliseconds for --allocator-residualize-base-low-sections. Defaults to the legacy fade-protect value when omitted.",
    )
    parser.add_argument(
        "--allocator-base-low-section-post-protect-ms",
        type=float,
        default=None,
        help="Optional post-roll protect width in milliseconds for --allocator-residualize-base-low-sections. Defaults to the legacy fade-protect value when omitted.",
    )
    parser.add_argument(
        "--allocator-base-low-section-transition-db",
        type=float,
        default=6.0,
        help="Soft reclaim gate transition width in dB used by --allocator-residualize-base-low-sections. Default: 6.0",
    )
    parser.add_argument(
        "--allocator-base-low-section-min-active-ms",
        type=float,
        default=0.0,
        help="Discard active islands shorter than this many milliseconds before applying reclaim protect. Default: 0.0",
    )
    parser.add_argument(
        "--allocator-base-low-section-gap-fill-ms",
        type=float,
        default=12.0,
        help="Fill inactive gaps shorter than this many milliseconds before protect is applied. Helps stabilize shorter RMS windows. Default: 12.0",
    )
    parser.add_argument(
        "--allocator-base-low-section-inactive-conf-scale",
        type=float,
        default=0.2,
        help="Scale factor for ownership confidence rescue inside inactive sections. Lower values suppress quiet spikes more aggressively. Default: 0.2",
    )
    parser.add_argument(
        "--allocator-base-low-section-inactive-keep-max",
        type=float,
        default=0.15,
        help="Hard upper cap for keep gate inside inactive sections after confidence rescue. Lower values suppress remaining quiet spikes more aggressively. Default: 0.15",
    )
    parser.add_argument(
        "--allocator-base-low-section-protect-floor",
        type=float,
        default=0.25,
        help="Minimum keep gate applied only inside pre/post protect halo (not fully active regions). Lower values reduce preserved quiet spikes. Default: 0.25",
    )
    parser.add_argument(
        "--allocator-mask-silent-sections",
        action="store_true",
        help="During allocator refinement, exclude stems from local residual routing where their short-time RMS is below the configured threshold.",
    )
    parser.add_argument(
        "--allocator-mask-silent-threshold-db",
        type=float,
        default=-40.0,
        help="Short-time RMS threshold in dBFS used by --allocator-mask-silent-sections. Default: -40.0",
    )
    parser.add_argument(
        "--allocator-mask-silent-window-ms",
        type=float,
        default=80.0,
        help="Sliding RMS window in milliseconds used by --allocator-mask-silent-sections. Default: 80.0",
    )
    parser.add_argument(
        "--allocator-guard-bass-high-closure",
        action="store_true",
        help=(
            "After exact mix closure, move only very quiet high-frequency closure residual out of bass "
            "when the base/pre-closure bass has no high-frequency evidence. Keeps the sum closed."
        ),
    )
    parser.add_argument(
        "--allocator-bass-high-closure-cutoff-hz",
        type=float,
        default=18000.0,
        help="High-pass cutoff for --allocator-guard-bass-high-closure. Default: 18000.",
    )
    parser.add_argument(
        "--allocator-bass-high-closure-transition-hz",
        type=float,
        default=1000.0,
        help="Smooth high-pass transition width for --allocator-guard-bass-high-closure. Default: 1000.",
    )
    parser.add_argument(
        "--allocator-bass-high-closure-protect-threshold-db",
        type=float,
        default=-95.0,
        help=(
            "Do not guard bass if base/pre-closure bass high-band local RMS is above this dBFS threshold. "
            "Lower is safer for slap/attack protection. Default: -95."
        ),
    )
    parser.add_argument(
        "--allocator-bass-high-closure-max-residual-db",
        type=float,
        default=-65.0,
        help=(
            "Only guard closure high-band residual whose local RMS is below this dBFS threshold. "
            "This limits the guard to quiet noise-floor material. Default: -65."
        ),
    )
    parser.add_argument(
        "--allocator-bass-high-closure-window-ms",
        type=float,
        default=80.0,
        help="Local RMS window for --allocator-guard-bass-high-closure. Default: 80.",
    )
    parser.add_argument(
        "--allocator-bass-high-closure-gate-transition-db",
        type=float,
        default=12.0,
        help="Soft gate transition width for --allocator-guard-bass-high-closure. Default: 12.",
    )
    parser.add_argument(
        "--silent-stem-threshold-db",
        type=float,
        default=-100.0,
        help="Peak threshold in dBFS used by --skip-silent-stems. Default: -100.0",
    )
    parser.add_argument(
        "--prefer-flac",
        action="store_true",
        help="Use FLAC when peak <= 1 and subtype supports it. Otherwise WAV is used.",
    )
    parser.add_argument(
        "--pcm-subtype",
        choices=["PCM_16", "PCM_24", "FLOAT"],
        default="FLOAT",
        help="Subtype for written audio files. Default: FLOAT",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable chunk progress bars.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def ensure_input_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if not path.is_file() and not path.is_dir():
        raise FileNotFoundError(f"Input must be a file or directory: {path}")


def parse_input_extensions(raw_extensions: str) -> set[str]:
    extensions: set[str] = set()
    for item in str(raw_extensions).split(","):
        item = item.strip().lower()
        if not item:
            continue
        extensions.add(item if item.startswith(".") else f".{item}")
    if not extensions:
        raise ValueError("--input-extensions must contain at least one extension.")
    return extensions


def iter_input_files(input_path: Path, *, recursive: bool, extensions: set[str]) -> list[Path]:
    if input_path.is_file():
        return [input_path.resolve()]

    pattern = "**/*" if recursive else "*"
    files = [
        path.resolve()
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(files, key=lambda path: str(path).lower())


def get_batch_relative_stem(input_root: Path, input_path: Path) -> Path:
    relative_path = input_path.resolve().relative_to(input_root.resolve())
    return relative_path.with_suffix("")


def get_output_root(args: argparse.Namespace) -> Path:
    if bool(getattr(args, "output_dir_is_stem_dir", False)):
        return args.output
    batch_relative_stem = getattr(args, "_batch_relative_stem", None)
    if batch_relative_stem is not None:
        return args.output / batch_relative_stem
    return args.output / args.input.stem


def resolve_device(device_name: str, cuda_device: int) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device(f"cuda:{cuda_device}")
    if device_name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this environment.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_device}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_audio(input_path: Path, sample_rate: int, expected_channels: int | None) -> tuple[np.ndarray, int]:
    mix, sr = librosa.load(str(input_path), sr=sample_rate, mono=False)
    if mix.ndim == 1:
        mix = np.expand_dims(mix, axis=0)
    if expected_channels == 2 and mix.shape[0] == 1:
        mix = np.concatenate([mix, mix], axis=0)
    return mix.astype(np.float32, copy=False), int(sr)


def normalize_audio(audio: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    mono = audio.mean(0)
    mean = float(mono.mean())
    std = float(mono.std())
    if std == 0.0:
        return audio, {"mean": mean, "std": 1.0}
    return (audio - mean) / std, {"mean": mean, "std": std}


def denormalize_audio(audio: np.ndarray, norm_params: dict[str, float]) -> np.ndarray:
    return audio * norm_params["std"] + norm_params["mean"]


def scale_difference_waveform_if_needed(
    audio: np.ndarray,
    config: ConfigDict,
    norm_params: dict[str, float] | None,
) -> np.ndarray:
    if norm_params is None:
        return audio
    if not bool(get_inference_setting(config, "normalize", False)):
        return audio
    return audio * norm_params["std"]


def safe_torch_load(*args, **kwargs):
    gelu = getattr(torch._C._nn, "gelu", None)
    if gelu is not None and hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals([gelu]):
            return torch.load(*args, **kwargs)
    return torch.load(*args, **kwargs)


def load_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors checkpoints requires safetensors. "
                "Install it with `pip install safetensors`."
            ) from exc
        device = str(map_location) if isinstance(map_location, (str, torch.device)) else "cpu"
        return load_file(str(checkpoint_path), device=device)

    return safe_torch_load(
        checkpoint_path,
        weights_only=weights_only,
        map_location=map_location,
    )


def load_yaml_config(path: Path) -> ConfigDict:
    with path.open("r", encoding="utf-8") as handle:
        return ConfigDict(yaml.load(handle, Loader=yaml.FullLoader))


def load_allocator_run_config(
    *,
    allocator_config_path: Path,
    allocator_checkpoint: dict[str, Any],
) -> OmegaConf:
    if allocator_config_path.is_file():
        return OmegaConf.load(allocator_config_path)

    run_config = allocator_checkpoint.get("run_config", None)
    if run_config is not None:
        return OmegaConf.create(run_config)

    raise ValueError(
        "Allocator config not found and checkpoint does not contain run_config. "
        f"Expected sidecar YAML at: {allocator_config_path}"
    )


def get_model_from_config(config_path: Path) -> tuple[torch.nn.Module, ConfigDict]:
    config = load_yaml_config(config_path)
    model_type = str(getattr(config.training, "model_type", "bs_roformer"))
    if model_type != "bs_roformer":
        raise ValueError(
            f"This standalone repo supports only bs_roformer configs, got: {model_type}"
        )
    model = BSRoformer(**dict(config.model))
    return model, config


def get_inference_setting(config: ConfigDict, key: str, default: Any = None) -> Any:
    inference = getattr(config, "inference", None)
    if inference is None:
        return default
    try:
        if hasattr(inference, "get"):
            return inference.get(key, default)
        return getattr(inference, key, default)
    except Exception:
        return default


def prefer_target_instrument(config: ConfigDict) -> list[str]:
    target = getattr(config.training, "target_instrument", None)
    if target:
        return [target]
    return list(config.training.instruments)


def get_stem_chunk_size_overrides(config: ConfigDict) -> dict[str, int]:
    raw_overrides = get_inference_setting(config, "stem_chunk_sizes", None)
    if not raw_overrides:
        return {}
    if not hasattr(raw_overrides, "items"):
        raise ValueError("inference.stem_chunk_sizes must be a mapping from stem name to chunk size.")

    instruments = set(prefer_target_instrument(config))
    overrides: dict[str, int] = {}
    for stem_name, chunk_size in raw_overrides.items():
        stem_name = str(stem_name)
        if stem_name not in instruments:
            raise ValueError(
                f"inference.stem_chunk_sizes contains unknown stem '{stem_name}'. "
                f"Known stems: {sorted(instruments)}"
            )
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError(f"inference.stem_chunk_sizes.{stem_name} must be positive.")
        overrides[stem_name] = chunk_size
    return overrides


def maybe_add_complementary_stem(
    waveforms: dict[str, np.ndarray],
    mix: np.ndarray,
    config: ConfigDict,
) -> tuple[dict[str, np.ndarray], list[str]]:
    configured_instruments = list(getattr(config.training, "instruments", []))
    target_instrument = getattr(config.training, "target_instrument", None)
    if len(configured_instruments) != 2 or not target_instrument or target_instrument not in waveforms:
        return waveforms, list(waveforms.keys())

    missing = [instr for instr in configured_instruments if instr not in waveforms]
    if len(missing) != 1:
        return waveforms, list(waveforms.keys())

    complement = missing[0]
    updated = dict(waveforms)
    updated[complement] = mix - updated[target_instrument]
    ordered = [instr for instr in configured_instruments if instr in updated]
    return updated, ordered


def maybe_add_derived_instrumentals(
    waveforms: dict[str, np.ndarray],
    mix: np.ndarray,
    ordered_instruments: list[str],
    *,
    extract_instrumental: bool,
    save_instrumental_variants: bool,
    save_mix_minus_instrumental_sum_non_vocals: bool,
) -> tuple[dict[str, np.ndarray], list[str]]:
    waveforms = dict(waveforms)
    ordered = list(ordered_instruments)

    def add_output(name: str, audio: np.ndarray) -> None:
        waveforms[name] = audio
        if name not in ordered:
            ordered.append(name)

    if extract_instrumental and "vocals" in waveforms:
        add_output("instrumental", mix - waveforms["vocals"])

    if save_instrumental_variants and "vocals" in waveforms:
        non_vocals = [instr for instr in ordered if instr != "vocals"]
        if non_vocals:
            sum_non_vocals = np.sum([waveforms[instr] for instr in non_vocals], axis=0)
            add_output("instrumental_sum_non_vocals", sum_non_vocals)
        add_output("instrumental_mix_minus_vocals", mix - waveforms["vocals"])

    if save_mix_minus_instrumental_sum_non_vocals and "vocals" in waveforms:
        non_vocals = [instr for instr in ordered if instr != "vocals"]
        if non_vocals:
            sum_non_vocals = np.sum([waveforms[instr] for instr in non_vocals], axis=0)
            add_output("mixture_minus_instrumental_sum_non_vocals", mix - sum_non_vocals)

    return waveforms, ordered


def maybe_add_collapsed_four_stem_from_six(
    waveforms: dict[str, np.ndarray],
    mix: np.ndarray,
    ordered_instruments: list[str],
    *,
    collapse_to_4stem: bool,
) -> tuple[dict[str, np.ndarray], list[str]]:
    if not collapse_to_4stem:
        return dict(waveforms), list(ordered_instruments)

    required = ("bass", "drums", "other", "vocals", "guitar", "piano")
    if not all(instr in waveforms for instr in required):
        print("Skipping 6stem->4stem collapse because required stems are not all available.")
        return dict(waveforms), list(ordered_instruments)

    updated = dict(waveforms)
    ordered = list(ordered_instruments)

    def add_output(name: str, audio: np.ndarray) -> None:
        updated[name] = audio
        if name not in ordered:
            ordered.append(name)

    bass_4stem = updated["bass"]
    drums_4stem = updated["drums"]
    vocals_4stem = updated["vocals"]
    other_4stem = updated["other"] + updated["guitar"] + updated["piano"]

    add_output("bass_4stem", bass_4stem)
    add_output("drums_4stem", drums_4stem)
    add_output("other_4stem", other_4stem)
    add_output("vocals_4stem", vocals_4stem)

    instrumental_sum = bass_4stem + drums_4stem + other_4stem
    add_output("instrumental_sum_non_vocals_4stem", instrumental_sum)
    add_output("instrumental_mix_minus_vocals_4stem", mix - vocals_4stem)
    add_output("mixture_minus_instrumental_sum_non_vocals_4stem", mix - instrumental_sum)

    return updated, ordered


def maybe_project_to_four_stem_plus_instrumental(
    waveforms: dict[str, np.ndarray],
    mix: np.ndarray,
    ordered_instruments: list[str],
    *,
    output_4stem_plus_instrumental: bool,
) -> tuple[dict[str, np.ndarray], list[str]]:
    if not output_4stem_plus_instrumental:
        return dict(waveforms), list(ordered_instruments)

    waveforms = dict(waveforms)

    def can_use(names: tuple[str, ...]) -> bool:
        return all(name in waveforms for name in names)

    selected = None
    if can_use(("bass_4stem", "drums_4stem", "other_4stem", "vocals_4stem")):
        selected = {
            "bass": waveforms["bass_4stem"],
            "drums": waveforms["drums_4stem"],
            "other": waveforms["other_4stem"],
            "vocals": waveforms["vocals_4stem"],
        }
    elif can_use(("bass", "drums", "other", "vocals", "guitar", "piano")):
        selected = {
            "bass": waveforms["bass"],
            "drums": waveforms["drums"],
            "other": waveforms["other"] + waveforms["guitar"] + waveforms["piano"],
            "vocals": waveforms["vocals"],
        }
    elif can_use(("bass", "drums", "other", "vocals")):
        selected = {
            "bass": waveforms["bass"],
            "drums": waveforms["drums"],
            "other": waveforms["other"],
            "vocals": waveforms["vocals"],
        }

    if selected is None:
        print("Skipping 4stem projection because required stems are not available.")
        return dict(waveforms), list(ordered_instruments)

    selected["instrumental"] = mix - selected["vocals"]
    return selected, ["bass", "drums", "other", "vocals", "instrumental"]


def normalize_checkpoint_state_dict(old_model: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "state" in old_model:
        old_model = old_model["state"]
    if "state_dict" in old_model:
        old_model = old_model["state_dict"]
    if "model_state_dict" in old_model:
        old_model = old_model["model_state_dict"]

    tensor_items: list[tuple[str, torch.Tensor]] = []
    module_prefix_count = 0
    for key, value in old_model.items():
        if not isinstance(key, str) or key.startswith("_") or not isinstance(value, torch.Tensor):
            continue
        if key == "n_averaged":
            continue
        if key.startswith("module."):
            module_prefix_count += 1
        tensor_items.append((key, value))

    strip_module_prefix = bool(tensor_items) and module_prefix_count == len(tensor_items)
    normalized: dict[str, torch.Tensor] = {}
    for key, value in tensor_items:
        if strip_module_prefix and key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value
    return normalized


def load_model_state(model: torch.nn.Module, checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
    state_dict = normalize_checkpoint_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as exc:
        filtered = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("pre_final_aux_mask_estimator.")
        }
        missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
        missing_keys = [
            key for key in missing_keys if not key.startswith("pre_final_aux_mask_estimator.")
        ]
        unexpected_keys = [
            key for key in unexpected_keys if not key.startswith("pre_final_aux_mask_estimator.")
        ]
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Failed to load checkpoint {checkpoint_path}: {exc}"
            ) from exc


def get_windowing_array(window_size: int, fade_size: int) -> torch.Tensor:
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def get_model_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def normalize_stem_weights(
    weights: torch.Tensor,
    fallback: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    weights_sum = weights.sum(dim=1, keepdim=True)
    fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(eps)
    normalized = weights / weights_sum.clamp_min(eps)
    return torch.where(weights_sum > eps, normalized, fallback)


def get_deterministic_simple_stem_names(config: ConfigDict, num_stems: int) -> list[str]:
    stem_names = list(prefer_target_instrument(config))
    if len(stem_names) == num_stems:
        return stem_names

    training_stems = list(getattr(config.training, "instruments", []))
    if len(training_stems) == num_stems:
        return training_stems

    if len(stem_names) > num_stems:
        return stem_names[:num_stems]
    return stem_names + [f"stem_{idx}" for idx in range(len(stem_names), num_stems)]


def get_deterministic_simple_lowfreq_gain(config: ConfigDict, stem_name: str) -> float:
    stem_key = str(stem_name).lower().replace("-", "_").replace(" ", "_")
    override_name = f"deterministic_simple_lowfreq_prior_{stem_key}"
    override_value = get_inference_setting(config, override_name, None)
    if override_value is not None:
        return float(override_value)

    defaults = {
        "bass": 1.30,
        "drums": 1.10,
        "vocals": 0.35,
        "other": 0.70,
        "guitar": 0.60,
        "piano": 0.60,
        "instrum": 1.05,
        "instrumental": 1.05,
    }
    return float(
        defaults.get(
            stem_key,
            get_inference_setting(config, "deterministic_simple_lowfreq_prior_default", 1.0),
        )
    )


def build_deterministic_simple_lowfreq_prior(
    config: ConfigDict,
    model_module: torch.nn.Module,
    stem_names: list[str],
    stem_mag: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    audio_channels = int(
        getattr(model_module, "audio_channels", getattr(config.audio, "num_channels", 2))
    )
    total_bins = int(stem_mag.shape[2])
    if total_bins % audio_channels != 0:
        return torch.ones((1, len(stem_names), total_bins, 1), device=device, dtype=stem_mag.dtype)

    num_freq_bins = total_bins // audio_channels
    sample_rate = float(getattr(config.audio, "sample_rate", 44100))
    n_fft = int(getattr(model_module, "stft_n_fft", getattr(config.audio, "n_fft", 2048)))
    cutoff_hz = float(
        get_inference_setting(config, "deterministic_simple_lowfreq_prior_cutoff_hz", 220.0)
    )
    slope = float(
        get_inference_setting(config, "deterministic_simple_lowfreq_prior_slope", 4.0)
    )
    if cutoff_hz <= 0.0:
        return torch.ones((1, len(stem_names), total_bins, 1), device=device, dtype=stem_mag.dtype)

    freq_hz = torch.arange(num_freq_bins, device=device, dtype=torch.float32)
    freq_hz = freq_hz * (sample_rate / float(n_fft))
    shelf = 1.0 / (1.0 + (freq_hz / cutoff_hz).clamp_min(0.0).pow(slope))
    shelf = shelf.repeat_interleave(audio_channels).view(1, 1, total_bins, 1)
    shelf = shelf.to(dtype=stem_mag.dtype)

    stem_gains = torch.tensor(
        [get_deterministic_simple_lowfreq_gain(config, name) for name in stem_names],
        device=device,
        dtype=stem_mag.dtype,
    ).view(1, len(stem_names), 1, 1)
    return 1.0 + (stem_gains - 1.0) * shelf


def audio_to_flattened_stft(
    model_module: torch.nn.Module,
    audio: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    x_is_mps = device.type == "mps"
    stft_window = model_module.stft_window_fn(device=device)
    audio = audio.to(device=device, dtype=torch.float32)

    if audio.ndim == 3:
        batch_size, num_channels = audio.shape[:2]
        packed = rearrange(audio, "b s t -> (b s) t")
        num_stems = 1
    elif audio.ndim == 4:
        batch_size, num_stems, num_channels = audio.shape[:3]
        packed = rearrange(audio, "b n s t -> (b n s) t")
    else:
        raise ValueError(f"Unsupported audio rank for STFT conversion: {audio.shape}")

    try:
        stft = torch.stft(
            packed,
            **model_module.stft_kwargs,
            window=stft_window,
            return_complex=True,
        )
    except Exception:
        stft = torch.stft(
            packed.cpu() if x_is_mps else packed,
            **model_module.stft_kwargs,
            window=stft_window.cpu() if x_is_mps else stft_window,
            return_complex=True,
        ).to(device)

    if num_stems == 1:
        return rearrange(stft, "(b s) f t -> b 1 (f s) t", b=batch_size, s=num_channels)
    return rearrange(
        stft,
        "(b n s) f t -> b n (f s) t",
        b=batch_size,
        n=num_stems,
        s=num_channels,
    )


def stft_to_audio_from_components(
    model_module: torch.nn.Module,
    masked_stft: torch.Tensor,
    device: torch.device,
    audio_length: int,
) -> torch.Tensor:
    x_is_mps = device.type == "mps"
    batch_size, num_stems = masked_stft.shape[:2]
    masked_stft = rearrange(
        masked_stft,
        "b n (f s) t -> (b n s) f t",
        s=model_module.audio_channels,
    )
    if getattr(model_module, "zero_dc", False):
        masked_stft = masked_stft.index_fill(1, torch.tensor(0, device=device), 0.0)

    stft_window = model_module.stft_window_fn(device=device)
    try:
        recon = torch.istft(
            masked_stft,
            **model_module.stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=audio_length,
        )
    except Exception:
        recon = torch.istft(
            masked_stft.cpu() if x_is_mps else masked_stft,
            **model_module.stft_kwargs,
            window=stft_window.cpu() if x_is_mps else stft_window,
            return_complex=False,
            length=audio_length,
        ).to(device)

    return rearrange(
        recon,
        "(b n s) t -> b n s t",
        b=batch_size,
        n=num_stems,
        s=model_module.audio_channels,
    )


def apply_deterministic_simple_residual_addback(
    config: ConfigDict,
    model_module: torch.nn.Module,
    pred_audio: torch.Tensor,
    mix_audio: torch.Tensor,
    device: torch.device,
    audio_length: int,
) -> torch.Tensor:
    eps = 1e-8
    tau = float(
        get_inference_setting(
            config,
            "deterministic_simple_threshold",
            get_inference_setting(config, "deterministic_owned_threshold", 0.6),
        )
    )
    gamma = float(
        get_inference_setting(
            config,
            "deterministic_simple_gamma",
            get_inference_setting(config, "deterministic_owned_gamma", 2.0),
        )
    )
    scale = float(
        get_inference_setting(
            config,
            "deterministic_simple_residual_scale",
            get_inference_setting(config, "deterministic_owned_residual_scale", 1.0),
        )
    )
    use_lowfreq_prior = bool(
        get_inference_setting(config, "deterministic_simple_use_stem_lowfreq_prior", False)
    )

    stem_stft = audio_to_flattened_stft(model_module, pred_audio, device)
    mix_stft = audio_to_flattened_stft(model_module, mix_audio, device)
    stem_mag = stem_stft.abs()

    owned = stem_mag / stem_mag.sum(dim=1, keepdim=True).clamp_min(eps)
    peakiness = owned / owned.amax(dim=1, keepdim=True).clamp_min(eps)
    owned = owned * peakiness

    lowfreq_prior = None
    if use_lowfreq_prior:
        stem_names = get_deterministic_simple_stem_names(config, pred_audio.shape[1])
        lowfreq_prior = build_deterministic_simple_lowfreq_prior(
            config=config,
            model_module=model_module,
            stem_names=stem_names,
            stem_mag=stem_mag,
            device=device,
        )
        owned = owned * lowfreq_prior

    if tau > 0.0:
        owned = torch.where(owned >= tau, owned, torch.zeros_like(owned))

    fallback = stem_mag
    if lowfreq_prior is not None:
        fallback = fallback * lowfreq_prior
    fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(eps)
    weights = normalize_stem_weights(owned.pow(gamma), fallback, eps=eps)

    residual_stft = mix_stft - stem_stft.sum(dim=1, keepdim=True)
    final_stft = stem_stft + (weights.to(stem_stft.dtype) * residual_stft) * scale
    return stft_to_audio_from_components(
        model_module=model_module,
        masked_stft=final_stft,
        device=device,
        audio_length=audio_length,
    )


def demix_with_chunk_size(
    config: ConfigDict,
    model: torch.nn.Module,
    mix: np.ndarray,
    device: torch.device,
    *,
    show_progress: bool,
    chunk_size: int,
    progress_desc: str,
) -> dict[str, np.ndarray]:
    model_module = get_model_module(model)
    use_deterministic_simple_addback = bool(
        get_inference_setting(config, "use_deterministic_simple_residual_addback", False)
    ) and not bool(getattr(model_module, "use_owned_calibrator", False))

    mix_tensor = torch.as_tensor(mix, dtype=torch.float32)
    num_instruments = len(prefer_target_instrument(config))
    num_overlap = int(get_inference_setting(config, "num_overlap", 2))
    fade_size = chunk_size // 10
    step = max(1, chunk_size // max(1, num_overlap))
    border = chunk_size - step
    length_init = int(mix_tensor.shape[-1])
    windowing_array = get_windowing_array(chunk_size, fade_size)

    if length_init > 2 * border and border > 0:
        mix_tensor = torch.nn.functional.pad(mix_tensor, (border, border), mode="reflect")

    batch_size = int(get_inference_setting(config, "batch_size", 1))
    use_amp = bool(getattr(config.training, "use_amp", True)) and device.type == "cuda"
    autocast_context = torch.cuda.amp.autocast(enabled=use_amp)

    with autocast_context:
        with torch.inference_mode():
            req_shape = (num_instruments,) + tuple(mix_tensor.shape)
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)

            i = 0
            batch_data: list[torch.Tensor] = []
            batch_locations: list[tuple[int, int]] = []
            progress_bar = None
            if show_progress:
                progress_bar = tqdm(total=mix_tensor.shape[1], desc=progress_desc, leave=False)

            while i < mix_tensor.shape[1]:
                part = mix_tensor[:, i : i + chunk_size].to(device)
                chunk_len = int(part.shape[-1])
                pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
                part = torch.nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)

                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                if len(batch_data) >= batch_size or i >= mix_tensor.shape[1]:
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)
                    if use_deterministic_simple_addback:
                        x = apply_deterministic_simple_residual_addback(
                            config=config,
                            model_module=model_module,
                            pred_audio=x,
                            mix_audio=arr,
                            device=device,
                            audio_length=arr.shape[-1],
                        )

                    for batch_index, (start, seg_len) in enumerate(batch_locations):
                        window = windowing_array
                        if start == 0 or (start + seg_len) >= mix_tensor.shape[1]:
                            window = windowing_array.clone()
                            if start == 0:
                                window[:fade_size] = 1
                            if (start + seg_len) >= mix_tensor.shape[1]:
                                window[-fade_size:] = 1
                        result[..., start : start + seg_len] += (
                            x[batch_index, ..., :seg_len].cpu() * window[..., :seg_len]
                        )
                        counter[..., start : start + seg_len] += window[..., :seg_len]

                    batch_data.clear()
                    batch_locations.clear()

                if progress_bar is not None:
                    progress_bar.update(step)

            if progress_bar is not None:
                progress_bar.close()

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if length_init > 2 * border and border > 0:
        estimated_sources = estimated_sources[..., border:-border]

    instruments = prefer_target_instrument(config)
    return {
        instrument: estimated_sources[index]
        for index, instrument in enumerate(instruments)
    }


def demix(
    config: ConfigDict,
    model: torch.nn.Module,
    mix: np.ndarray,
    device: torch.device,
    *,
    show_progress: bool,
) -> dict[str, np.ndarray]:
    default_chunk_size = int(get_inference_setting(config, "chunk_size", config.audio.chunk_size))
    waveforms = demix_with_chunk_size(
        config=config,
        model=model,
        mix=mix,
        device=device,
        show_progress=show_progress,
        chunk_size=default_chunk_size,
        progress_desc=f"Base demix {default_chunk_size}",
    )

    overrides = get_stem_chunk_size_overrides(config)
    chunk_to_stems: dict[int, list[str]] = {}
    for stem_name, chunk_size in overrides.items():
        if chunk_size == default_chunk_size:
            continue
        chunk_to_stems.setdefault(chunk_size, []).append(stem_name)

    for chunk_size, stem_names in sorted(chunk_to_stems.items()):
        override_waveforms = demix_with_chunk_size(
            config=config,
            model=model,
            mix=mix,
            device=device,
            show_progress=show_progress,
            chunk_size=chunk_size,
            progress_desc=f"Base demix {chunk_size}",
        )
        for stem_name in stem_names:
            waveforms[stem_name] = override_waveforms[stem_name]

    return waveforms


def apply_tta(
    config: ConfigDict,
    model: torch.nn.Module,
    mix: np.ndarray,
    waveforms_orig: dict[str, np.ndarray],
    device: torch.device,
    *,
    show_progress: bool,
) -> dict[str, np.ndarray]:
    track_proc_list = [mix[::-1].copy(), -1.0 * mix.copy()]
    for index, augmented_mix in enumerate(track_proc_list):
        waveforms = demix(
            config=config,
            model=model,
            mix=augmented_mix,
            device=device,
            show_progress=show_progress,
        )
        for stem_name in waveforms:
            if index == 0:
                waveforms_orig[stem_name] += waveforms[stem_name][::-1].copy()
            else:
                waveforms_orig[stem_name] -= waveforms[stem_name]

    for stem_name in waveforms_orig:
        waveforms_orig[stem_name] /= len(track_proc_list) + 1
    return waveforms_orig


def build_allocator_from_run_config(
    run_cfg: OmegaConf,
    state_dict: dict[str, torch.Tensor],
    base_config: ConfigDict,
    device: torch.device,
) -> torch.nn.Module:
    base_condition_dim = int(getattr(base_config.model, "dim", 0)) * 2
    allocator_type = str(getattr(run_cfg.allocator, "allocator_type", "conv")).lower()
    if allocator_type != "conv":
        raise ValueError(
            f"Unsupported allocator_type '{allocator_type}'. "
            "This public package includes only ConvResidualAllocator."
        )

    allocator_kwargs = dict(
        hidden_channels=int(getattr(run_cfg.allocator, "hidden_channels", 16)),
        kernel_size=int(getattr(run_cfg.allocator, "kernel_size", 7)),
        router_type=str(getattr(run_cfg.allocator, "router_type", "conv")),
        base_weight_gamma=float(getattr(run_cfg.allocator, "base_weight_gamma", 2.0)),
        blend_init_bias=float(getattr(run_cfg.allocator, "blend_init_bias", -4.0)),
        blend_floor=float(getattr(run_cfg.allocator, "blend_floor", 0.0)),
        delta_scale=float(getattr(run_cfg.allocator, "delta_scale", 4.0)),
        residual_scale_init=float(getattr(run_cfg.allocator, "residual_scale_init", 1.0)),
        residual_scale_min=float(getattr(run_cfg.allocator, "residual_scale_min", 0.25)),
        residual_scale_max=float(getattr(run_cfg.allocator, "residual_scale_max", 2.0)),
        learn_residual_scale=bool(getattr(run_cfg.allocator, "learn_residual_scale", True)),
        exclude_silent_stems=bool(getattr(run_cfg.allocator, "exclude_silent_stems", True)),
        silent_stem_abs_thresh=float(getattr(run_cfg.allocator, "silent_stem_abs_thresh", 0.0)),
        silent_stem_rel_thresh=float(getattr(run_cfg.allocator, "silent_stem_rel_thresh", 0.01)),
        silent_stem_time_kernel_size=int(getattr(run_cfg.allocator, "silent_stem_time_kernel_size", 9)),
        use_exact_mix_closure=bool(getattr(run_cfg.allocator, "use_exact_mix_closure", False)),
        exact_mix_closure_topk=int(getattr(run_cfg.allocator, "exact_mix_closure_topk", 0)),
        screening_num_bands=int(getattr(run_cfg.allocator, "screening_num_bands", 64)),
        screening_num_frames=int(getattr(run_cfg.allocator, "screening_num_frames", 128)),
        screening_eval_num_bands=(
            None
            if getattr(run_cfg.allocator, "screening_eval_num_bands", None) is None
            else int(getattr(run_cfg.allocator, "screening_eval_num_bands"))
        ),
        screening_eval_num_frames=(
            None
            if getattr(run_cfg.allocator, "screening_eval_num_frames", None) is None
            else int(getattr(run_cfg.allocator, "screening_eval_num_frames"))
        ),
        screening_token_dim=int(getattr(run_cfg.allocator, "screening_token_dim", 32)),
        screening_heads=int(getattr(run_cfg.allocator, "screening_heads", 4)),
        screening_dim_head=int(getattr(run_cfg.allocator, "screening_dim_head", 16)),
        screening_dropout=float(getattr(run_cfg.allocator, "screening_dropout", 0.0)),
        screening_norm_values=bool(getattr(run_cfg.allocator, "screening_norm_values", False)),
        screening_tanh_norm=bool(getattr(run_cfg.allocator, "screening_tanh_norm", True)),
        screening_init_window=float(getattr(run_cfg.allocator, "screening_init_window", 64.0)),
        screening_init_relevance_width=float(
            getattr(run_cfg.allocator, "screening_init_relevance_width", 4.0)
        ),
        screening_init_scale=float(getattr(run_cfg.allocator, "screening_init_scale", 0.0)),
        screening_init_mode=str(getattr(run_cfg.allocator, "screening_init_mode", "legacy")),
        screening_random_std=float(getattr(run_cfg.allocator, "screening_random_std", 0.02)),
        use_small_delta_branch=bool(getattr(run_cfg.allocator, "use_small_delta_branch", False)),
        delta_branch_hidden_channels=(
            None
            if getattr(run_cfg.allocator, "delta_branch_hidden_channels", None) is None
            else int(getattr(run_cfg.allocator, "delta_branch_hidden_channels"))
        ),
        delta_branch_freq_kernel_size=int(
            getattr(run_cfg.allocator, "delta_branch_freq_kernel_size", 5)
        ),
        delta_branch_time_kernel_size=int(
            getattr(run_cfg.allocator, "delta_branch_time_kernel_size", 3)
        ),
        delta_branch_scale=float(getattr(run_cfg.allocator, "delta_branch_scale", 0.1)),
        use_context_conditioning=bool(getattr(run_cfg.allocator, "use_context_conditioning", False)),
        context_feature_dim=int(
            getattr(run_cfg.allocator, "context_feature_dim", base_condition_dim)
        ),
        condition_mode=str(getattr(run_cfg.allocator, "condition_mode", "mlp")),
        condition_hidden_dim=int(getattr(run_cfg.allocator, "condition_hidden_dim", 128)),
        condition_router_scale=float(getattr(run_cfg.allocator, "condition_router_scale", 0.5)),
        condition_delta_scale=float(getattr(run_cfg.allocator, "condition_delta_scale", 0.5)),
        condition_blend_scale=float(getattr(run_cfg.allocator, "condition_blend_scale", 0.25)),
        judge_num_latents=int(getattr(run_cfg.allocator, "judge_num_latents", 8)),
        judge_latent_dim=int(getattr(run_cfg.allocator, "judge_latent_dim", 128)),
        judge_heads=int(getattr(run_cfg.allocator, "judge_heads", 4)),
        judge_context_num_frames=int(getattr(run_cfg.allocator, "judge_context_num_frames", 16)),
        judge_context_num_bands=int(getattr(run_cfg.allocator, "judge_context_num_bands", 8)),
        judge_dropout=float(getattr(run_cfg.allocator, "judge_dropout", 0.0)),
        inactive_conf_scale=float(getattr(run_cfg.allocator, "inactive_conf_scale", 0.2)),
        inactive_keep_floor=float(getattr(run_cfg.allocator, "inactive_keep_floor", 0.25)),
        inactive_keep_max=float(getattr(run_cfg.allocator, "inactive_keep_max", 1.0)),
        use_artifact_detector=bool(getattr(run_cfg.allocator, "use_artifact_detector", False)),
        artifact_hidden_channels=(
            None
            if getattr(run_cfg.allocator, "artifact_hidden_channels", None) is None
            else int(getattr(run_cfg.allocator, "artifact_hidden_channels"))
        ),
        artifact_kernel_size=int(getattr(run_cfg.allocator, "artifact_kernel_size", 3)),
        artifact_init_bias=float(getattr(run_cfg.allocator, "artifact_init_bias", -6.0)),
        artifact_max_suppression=float(getattr(run_cfg.allocator, "artifact_max_suppression", 1.0)),
        artifact_keep_floor=float(getattr(run_cfg.allocator, "artifact_keep_floor", 0.0)),
        artifact_base_active_mix_ratio=float(getattr(run_cfg.allocator, "artifact_base_active_mix_ratio", 0.02)),
        artifact_gt_inactive_mix_ratio=float(getattr(run_cfg.allocator, "artifact_gt_inactive_mix_ratio", 0.006)),
        artifact_gt_active_mix_ratio=float(getattr(run_cfg.allocator, "artifact_gt_active_mix_ratio", 0.02)),
        artifact_over_gt_margin_db=float(getattr(run_cfg.allocator, "artifact_over_gt_margin_db", 12.0)),
    )
    allocator = ConvResidualAllocator(**allocator_kwargs)
    allocator.load_state_dict(upgrade_allocator_state_dict(state_dict))
    allocator = allocator.to(device)
    allocator.eval()
    return allocator


def refine_with_allocator_chunked(
    *,
    allocator: torch.nn.Module,
    base_model: torch.nn.Module,
    base_config: ConfigDict,
    mix: np.ndarray,
    base_waveforms: dict[str, np.ndarray],
    device: torch.device,
    show_progress: bool,
    collect_diagnostics: bool,
    prune_base_silent_stems: bool,
    base_silent_threshold_db: float,
    base_silent_window_ms: float,
    residualize_base_low_sections: bool,
    base_low_section_threshold_db: float,
    base_low_section_window_ms: float,
    base_low_section_fade_protect_ms: float,
    base_low_section_pre_protect_ms: float | None,
    base_low_section_post_protect_ms: float | None,
    base_low_section_transition_db: float,
    base_low_section_min_active_ms: float,
    base_low_section_gap_fill_ms: float,
    base_low_section_protect_floor: float,
    mask_silent_sections: bool,
    mask_silent_threshold_db: float,
    mask_silent_window_ms: float,
    guard_bass_high_closure: bool,
    bass_high_closure_cutoff_hz: float,
    bass_high_closure_transition_hz: float,
    bass_high_closure_protect_threshold_db: float,
    bass_high_closure_max_residual_db: float,
    bass_high_closure_window_ms: float,
    bass_high_closure_gate_transition_db: float,
) -> tuple[
    dict[str, np.ndarray],
    list[str],
    dict[str, dict[str, np.ndarray] | np.ndarray] | None,
    list[str],
]:
    ordered_instruments = list(prefer_target_instrument(base_config))
    missing = [instr for instr in ordered_instruments if instr not in base_waveforms]
    if missing:
        raise RuntimeError(f"Base demix did not produce required stems: {missing}")

    base_waveforms, pruned_base_silent_stems = maybe_prune_base_silent_stems_for_allocator(
        base_waveforms,
        ordered_instruments,
        sample_rate=int(getattr(base_config.audio, "sample_rate", 44100)),
        prune_base_silent_stems=prune_base_silent_stems,
        base_silent_threshold_db=base_silent_threshold_db,
        base_silent_window_ms=base_silent_window_ms,
    )

    base_pred_np = np.stack(
        [base_waveforms[instr] for instr in ordered_instruments],
        axis=0,
    ).astype(np.float32, copy=False)
    mix_np = np.asarray(mix, dtype=np.float32)
    total_length = int(mix_np.shape[-1])
    sample_rate = int(getattr(base_config.audio, "sample_rate", 44100))

    if mask_silent_sections:
        print(
            "Masking allocator-silent sections "
            f"(RMS <= {min(float(mask_silent_threshold_db), 0.0):.1f} dBFS / {float(mask_silent_window_ms):.1f} ms)."
        )
    if guard_bass_high_closure:
        print(
            "Guarding bass high-frequency closure residual "
            f"(cutoff {max(float(bass_high_closure_cutoff_hz), 0.0):.0f} Hz, "
            f"transition {max(float(bass_high_closure_transition_hz), 1.0):.0f} Hz, "
            f"protect <= {float(bass_high_closure_protect_threshold_db):.1f} dBFS, "
            f"residual <= {float(bass_high_closure_max_residual_db):.1f} dBFS)."
        )
    if residualize_base_low_sections:
        pre_protect_ms = (
            max(float(base_low_section_fade_protect_ms), 0.0)
            if base_low_section_pre_protect_ms is None
            else max(float(base_low_section_pre_protect_ms), 0.0)
        )
        post_protect_ms = (
            max(float(base_low_section_fade_protect_ms), 0.0)
            if base_low_section_post_protect_ms is None
            else max(float(base_low_section_post_protect_ms), 0.0)
        )
        print(
            "Reclaiming low-level base sections inside allocator "
            f"(RMS <= {min(float(base_low_section_threshold_db), 0.0):.1f} dBFS / {float(base_low_section_window_ms):.1f} ms, "
            f"pre {pre_protect_ms:.1f} ms / post {post_protect_ms:.1f} ms, "
            f"min active {max(float(base_low_section_min_active_ms), 0.0):.1f} ms, "
            f"gap fill {max(float(base_low_section_gap_fill_ms), 0.0):.1f} ms, "
            f"transition {max(float(base_low_section_transition_db), 0.0):.1f} dB, "
            f"protect floor {min(max(float(base_low_section_protect_floor), 0.0), 1.0):.2f})."
        )
        if any(is_vocal_stem_name(name) for name in ordered_instruments):
            print("Reclaimed instrument-origin sections will not be routed back into vocal stems.")

    chunk_size = int(
        get_inference_setting(
            base_config,
            "allocator_chunk_size",
            get_inference_setting(base_config, "chunk_size", base_config.audio.chunk_size),
        )
    )
    num_overlap = int(get_inference_setting(base_config, "num_overlap", 2))
    step = max(1, chunk_size // max(1, num_overlap))
    fade_size = max(1, chunk_size // 10)
    border = chunk_size - step

    if total_length > 2 * border and border > 0:
        mix_work = np.pad(mix_np, ((0, 0), (border, border)), mode="reflect")
        base_work = np.pad(base_pred_np, ((0, 0), (0, 0), (border, border)), mode="reflect")
        trim_border = True
    else:
        mix_work = mix_np
        base_work = base_pred_np
        trim_border = False

    result = np.zeros_like(base_work, dtype=np.float32)
    counter = np.zeros_like(base_work, dtype=np.float32)
    diagnostics = None
    residual_counter = None
    if collect_diagnostics:
        diagnostics = {
            "routed_residual": np.zeros_like(base_work, dtype=np.float32),
            "direct_delta": np.zeros_like(base_work, dtype=np.float32),
            "total_delta": np.zeros_like(base_work, dtype=np.float32),
            "closure_delta": np.zeros_like(base_work, dtype=np.float32),
            "bass_high_closure_guard_removed": np.zeros_like(base_work, dtype=np.float32),
            "bass_high_closure_guard_redistributed": np.zeros_like(base_work, dtype=np.float32),
            "base_low_removed_source": np.zeros_like(base_work, dtype=np.float32),
            "base_low_redistributed": np.zeros_like(base_work, dtype=np.float32),
            "stem_reclaim_keep_mask": np.zeros_like(base_work, dtype=np.float32),
            "stem_delta_gate": np.zeros_like(base_work, dtype=np.float32),
            "gate_confidence": np.zeros_like(base_work, dtype=np.float32),
            "stem_activity_mask": np.zeros_like(base_work, dtype=np.float32),
            "remaining_residual": np.zeros_like(mix_work, dtype=np.float32),
        }
        residual_counter = np.zeros_like(mix_work, dtype=np.float32)

    windowing_array = get_windowing_array(chunk_size, fade_size).cpu().numpy().astype(np.float32)
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(total=mix_work.shape[-1], desc="Allocator refine", leave=False)

    with torch.inference_mode():
        start = 0
        while start < mix_work.shape[-1]:
            mix_chunk = mix_work[:, start : start + chunk_size]
            pred_chunk = base_work[:, :, start : start + chunk_size]
            chunk_len = int(mix_chunk.shape[-1])

            if chunk_len < chunk_size:
                pad_width = chunk_size - chunk_len
                if chunk_len > chunk_size // 2:
                    mix_chunk = np.pad(mix_chunk, ((0, 0), (0, pad_width)), mode="reflect")
                    pred_chunk = np.pad(pred_chunk, ((0, 0), (0, 0), (0, pad_width)), mode="reflect")
                else:
                    mix_chunk = np.pad(mix_chunk, ((0, 0), (0, pad_width)), mode="constant")
                    pred_chunk = np.pad(pred_chunk, ((0, 0), (0, 0), (0, pad_width)), mode="constant")

            mix_tensor = torch.from_numpy(mix_chunk).unsqueeze(0).to(device=device, dtype=torch.float32)
            pred_tensor = torch.from_numpy(pred_chunk).unsqueeze(0).to(device=device, dtype=torch.float32)
            original_pred_tensor = pred_tensor
            stem_activity_mask_override = None
            stem_reclaim_mask_override = None
            stem_delta_gate_override = None
            base_low_removed_tensor = torch.zeros_like(pred_tensor)
            base_low_redistributed_tensor = torch.zeros_like(pred_tensor)
            reclaim_keep_mask_tensor = torch.zeros_like(pred_tensor)
            delta_gate_tensor = torch.zeros_like(pred_tensor)
            gate_confidence_tensor = torch.zeros_like(pred_tensor)
            activity_mask_tensor = torch.zeros_like(pred_tensor)
            if residualize_base_low_sections:
                base_low_keep_gate = build_allocator_local_activity_gate(
                    pred_tensor,
                    sample_rate=sample_rate,
                    threshold_db=base_low_section_threshold_db,
                    window_ms=base_low_section_window_ms,
                    protect_ms=base_low_section_fade_protect_ms,
                    pre_protect_ms=base_low_section_pre_protect_ms,
                    post_protect_ms=base_low_section_post_protect_ms,
                    min_active_ms=base_low_section_min_active_ms,
                    gap_fill_ms=base_low_section_gap_fill_ms,
                    transition_db=base_low_section_transition_db,
                    protect_floor=base_low_section_protect_floor,
                )
                base_low_keep_mask = build_allocator_local_activity_mask(
                    pred_tensor,
                    sample_rate=sample_rate,
                    threshold_db=base_low_section_threshold_db,
                    window_ms=base_low_section_window_ms,
                    protect_ms=base_low_section_fade_protect_ms,
                    pre_protect_ms=base_low_section_pre_protect_ms,
                    post_protect_ms=base_low_section_post_protect_ms,
                    min_active_ms=base_low_section_min_active_ms,
                    gap_fill_ms=base_low_section_gap_fill_ms,
                )
                stem_reclaim_mask_override = base_low_keep_gate
                stem_delta_gate_override = base_low_keep_gate
                stem_activity_mask_override = base_low_keep_mask
            if mask_silent_sections:
                section_keep_mask = build_allocator_local_activity_mask(
                    pred_tensor,
                    sample_rate=sample_rate,
                    threshold_db=mask_silent_threshold_db,
                    window_ms=mask_silent_window_ms,
                )
                if stem_activity_mask_override is None:
                    stem_activity_mask_override = section_keep_mask
                else:
                    stem_activity_mask_override = stem_activity_mask_override & section_keep_mask
            refined_tensor, aux = allocator(
                base_model,
                mix_tensor,
                pred_tensor,
                device,
                stem_activity_mask_override=stem_activity_mask_override,
                stem_reclaim_mask_override=stem_reclaim_mask_override,
                stem_delta_gate_override=stem_delta_gate_override,
            )
            pre_closure_tensor = aux.get("pre_closure_audio", refined_tensor)
            bass_high_guard_removed_tensor = torch.zeros_like(refined_tensor)
            bass_high_guard_redistributed_tensor = torch.zeros_like(refined_tensor)
            if guard_bass_high_closure:
                refined_tensor, bass_high_guard_removed_tensor, bass_high_guard_redistributed_tensor = (
                    apply_bass_high_closure_guard(
                        refined_tensor,
                        pre_closure_tensor,
                        original_pred_tensor,
                        ordered_instruments,
                        sample_rate=sample_rate,
                        cutoff_hz=bass_high_closure_cutoff_hz,
                        transition_hz=bass_high_closure_transition_hz,
                        protect_threshold_db=bass_high_closure_protect_threshold_db,
                        max_residual_db=bass_high_closure_max_residual_db,
                        window_ms=bass_high_closure_window_ms,
                        gate_transition_db=bass_high_closure_gate_transition_db,
                    )
                )
            refined_chunk = refined_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)

            if diagnostics is not None:
                reclaimed_stft = aux.get("reclaimed_stft", None)
                reclaimed_residual_stft = aux.get("reclaimed_residual_stft", None)
                routed_residual_stft = (
                    aux["residual_scale"].to(
                        device=aux["residual_stft"].device,
                        dtype=aux["weights"].dtype,
                    )
                    * aux["weights"].to(dtype=aux["residual_stft"].dtype)
                    * aux["residual_stft"]
                )
                routed_reclaimed_stft = None
                if isinstance(reclaimed_residual_stft, torch.Tensor):
                    routed_reclaimed_stft = (
                        aux["residual_scale"].to(
                            device=reclaimed_residual_stft.device,
                            dtype=aux["weights"].dtype,
                        )
                        * aux["weights"].to(dtype=reclaimed_residual_stft.dtype)
                        * reclaimed_residual_stft
                    )
                routed_residual_audio = allocator._flattened_stft_to_audio(
                    base_model,
                    routed_residual_stft,
                    device,
                    mix_tensor.shape[-1],
                )
                if isinstance(reclaimed_stft, torch.Tensor):
                    reclaimed_audio = allocator._flattened_stft_to_audio(
                        base_model,
                        reclaimed_stft.to(dtype=aux["residual_stft"].dtype),
                        device,
                        mix_tensor.shape[-1],
                    )
                    base_low_removed_tensor = reclaimed_audio
                if routed_reclaimed_stft is not None:
                    reclaimed_redistributed_audio = allocator._flattened_stft_to_audio(
                        base_model,
                        routed_reclaimed_stft,
                        device,
                        mix_tensor.shape[-1],
                    )
                    base_low_redistributed_tensor = reclaimed_redistributed_audio
                direct_delta_audio = allocator._flattened_stft_to_audio(
                    base_model,
                    aux["direct_delta_stft"].to(dtype=aux["residual_stft"].dtype),
                    device,
                    mix_tensor.shape[-1],
                )
                reclaim_keep_np = control_tensor_to_audio_like(
                    aux.get("stem_reclaim_keep_mask", None),
                    num_channels=pred_tensor.shape[2],
                    target_num_samples=pred_tensor.shape[-1],
                )
                if reclaim_keep_np is not None:
                    reclaim_keep_mask_tensor = torch.from_numpy(reclaim_keep_np).unsqueeze(0).to(
                        device=pred_tensor.device,
                        dtype=pred_tensor.dtype,
                    )
                delta_gate_np = control_tensor_to_audio_like(
                    aux.get("stem_delta_gate", None),
                    num_channels=pred_tensor.shape[2],
                    target_num_samples=pred_tensor.shape[-1],
                )
                if delta_gate_np is not None:
                    delta_gate_tensor = torch.from_numpy(delta_gate_np).unsqueeze(0).to(
                        device=pred_tensor.device,
                        dtype=pred_tensor.dtype,
                    )
                gate_confidence_np = control_tensor_to_audio_like(
                    aux.get("gate_confidence", None),
                    num_channels=pred_tensor.shape[2],
                    target_num_samples=pred_tensor.shape[-1],
                )
                if gate_confidence_np is not None:
                    gate_confidence_tensor = torch.from_numpy(gate_confidence_np).unsqueeze(0).to(
                        device=pred_tensor.device,
                        dtype=pred_tensor.dtype,
                    )
                activity_mask_np = control_tensor_to_audio_like(
                    aux.get("stem_activity_mask", None),
                    num_channels=pred_tensor.shape[2],
                    target_num_samples=pred_tensor.shape[-1],
                )
                if activity_mask_np is not None:
                    activity_mask_tensor = torch.from_numpy(activity_mask_np).unsqueeze(0).to(
                        device=pred_tensor.device,
                        dtype=pred_tensor.dtype,
                    )
                total_delta_audio = refined_tensor - original_pred_tensor
                closure_reference_tensor = pre_closure_tensor
                closure_delta_audio = refined_tensor - closure_reference_tensor
                remaining_residual_audio = mix_tensor - refined_tensor.sum(dim=1)

                routed_residual_chunk = (
                    routed_residual_audio.squeeze(0).detach().cpu().numpy().astype(np.float32)
                )
                direct_delta_chunk = (
                    direct_delta_audio.squeeze(0).detach().cpu().numpy().astype(np.float32)
                )
                total_delta_chunk = total_delta_audio.squeeze(0).detach().cpu().numpy().astype(np.float32)
                closure_delta_chunk = (
                    closure_delta_audio.squeeze(0).detach().cpu().numpy().astype(np.float32)
                )
                bass_high_guard_removed_chunk = (
                    bass_high_guard_removed_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
                )
                bass_high_guard_redistributed_chunk = (
                    bass_high_guard_redistributed_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
                )
                remaining_residual_chunk = (
                    remaining_residual_audio.squeeze(0).detach().cpu().numpy().astype(np.float32)
                )
                base_low_removed_chunk = (
                    base_low_removed_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
                )
                base_low_redistributed_chunk = (
                    base_low_redistributed_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
                )
                reclaim_keep_chunk = (
                    reclaim_keep_mask_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
                )
                delta_gate_chunk = (
                    delta_gate_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
                )
                gate_confidence_chunk = (
                    gate_confidence_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
                )
                activity_mask_chunk = (
                    activity_mask_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
                )

            window = windowing_array
            if start == 0 or (start + chunk_len) >= mix_work.shape[-1]:
                window = windowing_array.copy()
                if start == 0:
                    window[:fade_size] = 1.0
                if (start + chunk_len) >= mix_work.shape[-1]:
                    window[-fade_size:] = 1.0

            result[:, :, start : start + chunk_len] += refined_chunk[:, :, :chunk_len] * window[:chunk_len]
            counter[:, :, start : start + chunk_len] += window[:chunk_len]

            if diagnostics is not None:
                diagnostics["routed_residual"][:, :, start : start + chunk_len] += (
                    routed_residual_chunk[:, :, :chunk_len] * window[:chunk_len]
                )
                diagnostics["direct_delta"][:, :, start : start + chunk_len] += (
                    direct_delta_chunk[:, :, :chunk_len] * window[:chunk_len]
                )
                diagnostics["total_delta"][:, :, start : start + chunk_len] += (
                    total_delta_chunk[:, :, :chunk_len] * window[:chunk_len]
                )
                diagnostics["closure_delta"][:, :, start : start + chunk_len] += (
                    closure_delta_chunk[:, :, :chunk_len] * window[:chunk_len]
                )
                diagnostics["bass_high_closure_guard_removed"][:, :, start : start + chunk_len] += (
                    bass_high_guard_removed_chunk[:, :, :chunk_len] * window[:chunk_len]
                )
                diagnostics["bass_high_closure_guard_redistributed"][:, :, start : start + chunk_len] += (
                    bass_high_guard_redistributed_chunk[:, :, :chunk_len] * window[:chunk_len]
                )
                diagnostics["base_low_removed_source"][:, :, start : start + chunk_len] += (
                    base_low_removed_chunk[:, :, :chunk_len] * window[:chunk_len]
                )
                diagnostics["base_low_redistributed"][:, :, start : start + chunk_len] += (
                    base_low_redistributed_chunk[:, :, :chunk_len] * window[:chunk_len]
                )
                diagnostics["stem_reclaim_keep_mask"][:, :, start : start + chunk_len] += (
                    reclaim_keep_chunk[:, :, :chunk_len] * window[:chunk_len]
                )
                diagnostics["stem_delta_gate"][:, :, start : start + chunk_len] += (
                    delta_gate_chunk[:, :, :chunk_len] * window[:chunk_len]
                )
                diagnostics["gate_confidence"][:, :, start : start + chunk_len] += (
                    gate_confidence_chunk[:, :, :chunk_len] * window[:chunk_len]
                )
                diagnostics["stem_activity_mask"][:, :, start : start + chunk_len] += (
                    activity_mask_chunk[:, :, :chunk_len] * window[:chunk_len]
                )
                diagnostics["remaining_residual"][:, start : start + chunk_len] += (
                    remaining_residual_chunk[:, :chunk_len] * window[:chunk_len]
                )
                residual_counter[:, start : start + chunk_len] += window[:chunk_len]

            del mix_tensor, original_pred_tensor, pred_tensor, refined_tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()

            start += step
            if progress_bar is not None:
                progress_bar.update(step)

    if progress_bar is not None:
        progress_bar.close()

    refined_np = result / np.clip(counter, a_min=1e-8, a_max=None)
    np.nan_to_num(refined_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if trim_border:
        refined_np = refined_np[..., border:-border]

    refined_waveforms = {
        instr: refined_np[index, :, :total_length]
        for index, instr in enumerate(ordered_instruments)
    }

    diagnostic_outputs = None
    if diagnostics is not None:
        stem_counter = np.clip(counter, a_min=1e-8, a_max=None)
        residual_counter = np.clip(residual_counter, a_min=1e-8, a_max=None)
        diagnostic_outputs = {}
        for name in (
            "routed_residual",
            "direct_delta",
            "total_delta",
            "closure_delta",
            "bass_high_closure_guard_removed",
            "bass_high_closure_guard_redistributed",
            "base_low_removed_source",
            "base_low_redistributed",
            "stem_reclaim_keep_mask",
            "stem_delta_gate",
            "gate_confidence",
            "stem_activity_mask",
        ):
            diag_np = diagnostics[name] / stem_counter
            np.nan_to_num(diag_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if trim_border:
                diag_np = diag_np[..., border:-border]
            diagnostic_outputs[name] = {
                instr: diag_np[index, :, :total_length]
                for index, instr in enumerate(ordered_instruments)
            }

        remaining_residual_np = diagnostics["remaining_residual"] / residual_counter
        np.nan_to_num(remaining_residual_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if trim_border:
            remaining_residual_np = remaining_residual_np[..., border:-border]
        diagnostic_outputs["remaining_residual"] = remaining_residual_np[:, :total_length]

    return refined_waveforms, ordered_instruments, diagnostic_outputs, pruned_base_silent_stems


def split_signed_waveform(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sum_part = np.where(audio > 0.0, audio, 0.0)
    sub_part = np.where(audio < 0.0, audio, 0.0)
    return (
        sum_part.astype(np.float32, copy=False),
        sub_part.astype(np.float32, copy=False),
    )


def control_tensor_to_audio_like(
    control: torch.Tensor | None,
    *,
    num_channels: int,
    target_num_samples: int | None = None,
) -> np.ndarray | None:
    if not isinstance(control, torch.Tensor):
        return None
    if control.ndim != 4 or control.shape[2] != 1:
        raise ValueError(f"Expected control tensor with shape [b, n, 1, t], got {tuple(control.shape)}")
    control = control.detach().to(dtype=torch.float32)
    if target_num_samples is not None and control.shape[-1] != target_num_samples:
        batch_size, num_stems, _, num_frames = control.shape
        control = control.reshape(batch_size * num_stems, 1, num_frames)
        control = F.interpolate(
            control,
            size=target_num_samples,
            mode="linear",
            align_corners=False,
        )
        control = control.reshape(batch_size, num_stems, 1, target_num_samples)
    control = control.expand(-1, -1, num_channels, -1)
    return control.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def choose_codec_and_subtype(audio: np.ndarray, args: argparse.Namespace) -> tuple[str, str]:
    peak = float(np.abs(audio).max())
    codec = "flac" if args.prefer_flac and peak <= 1.0 and args.pcm_subtype != "FLOAT" else "wav"
    subtype = args.pcm_subtype
    if subtype in sf.available_subtypes(codec):
        return codec, subtype
    return codec, sf.default_subtype(codec)


def save_audio_file(
    path_without_suffix: Path,
    audio: np.ndarray,
    sample_rate: int,
    args: argparse.Namespace,
) -> Path:
    codec, subtype = choose_codec_and_subtype(audio, args)
    output_path = path_without_suffix.with_suffix(f".{codec}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio.T, sample_rate, subtype=subtype)
    return output_path


def waveform_peak(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0))))


def dbfs_to_amplitude(dbfs: float) -> float:
    value = float(dbfs)
    if not math.isfinite(value):
        return 0.0 if value < 0.0 else float("inf")
    value = min(value, 0.0)
    if value <= -200.0:
        return 0.0
    return float(10.0 ** (value / 20.0))


def waveform_max_rms(audio: np.ndarray, window_samples: int) -> float:
    safe_audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if safe_audio.size == 0:
        return 0.0

    if safe_audio.ndim == 1:
        power = np.square(safe_audio, dtype=np.float32)
    else:
        power = np.mean(np.square(safe_audio, dtype=np.float32), axis=0)

    if power.size == 0:
        return 0.0

    window = max(1, min(int(window_samples), int(power.size)))
    if window == 1:
        return float(np.sqrt(np.max(power)))

    cumsum = np.cumsum(np.pad(power.astype(np.float64, copy=False), (1, 0)))
    window_mean = (cumsum[window:] - cumsum[:-window]) / float(window)
    return float(np.sqrt(np.max(window_mean))) if window_mean.size else 0.0


def _remove_short_active_runs(active: torch.Tensor, min_active_samples: int) -> torch.Tensor:
    if min_active_samples <= 1:
        return active

    flat_active = active.squeeze(1).to(dtype=torch.bool)
    cleaned = flat_active.clone()
    row_count = int(flat_active.shape[0])
    for row_idx in range(row_count):
        row = flat_active[row_idx]
        padded = F.pad(row.to(dtype=torch.int8), (1, 1))
        changes = padded[1:] - padded[:-1]
        starts = torch.nonzero(changes == 1, as_tuple=False).flatten()
        ends = torch.nonzero(changes == -1, as_tuple=False).flatten()
        if starts.numel() == 0:
            continue
        for start, end in zip(starts.tolist(), ends.tolist()):
            if end - start < min_active_samples:
                cleaned[row_idx, start:end] = False
    return cleaned.unsqueeze(1)


def _fill_short_inactive_gaps(active: torch.Tensor, max_gap_samples: int) -> torch.Tensor:
    if max_gap_samples <= 0:
        return active

    filled = active.clone()
    flat_active = rearrange(active, "bn 1 t -> bn t")
    row_count = flat_active.shape[0]
    for row_idx in range(row_count):
        row = flat_active[row_idx]
        inactive = ~row
        padded = F.pad(inactive.to(dtype=torch.int8), (1, 1))
        changes = padded[1:] - padded[:-1]
        starts = torch.nonzero(changes == 1, as_tuple=False).flatten()
        ends = torch.nonzero(changes == -1, as_tuple=False).flatten()
        if starts.numel() == 0:
            continue
        for start, end in zip(starts.tolist(), ends.tolist()):
            gap_len = end - start
            if gap_len <= max_gap_samples:
                has_left_active = start > 0 and bool(row[start - 1])
                has_right_active = end < row.numel() and bool(row[end])
                if has_left_active and has_right_active:
                    filled[row_idx, 0, start:end] = True
    return filled


def _dilate_active_regions(
    active: torch.Tensor,
    *,
    pre_protect_samples: int,
    post_protect_samples: int,
) -> torch.Tensor:
    if pre_protect_samples <= 0 and post_protect_samples <= 0:
        return active

    kernel_size = int(pre_protect_samples + post_protect_samples + 1)
    padded = F.pad(
        active.to(dtype=torch.float32),
        (int(pre_protect_samples), int(post_protect_samples)),
        mode="constant",
        value=0.0,
    )
    dilated = F.max_pool1d(
        padded,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
    )
    return dilated > 0.5


def build_allocator_local_activity_mask(
    base_pred_audio: torch.Tensor,
    *,
    sample_rate: int,
    threshold_db: float,
    window_ms: float,
    protect_ms: float = 0.0,
    pre_protect_ms: float | None = None,
    post_protect_ms: float | None = None,
    min_active_ms: float = 0.0,
    gap_fill_ms: float = 0.0,
) -> torch.Tensor:
    safe_audio = torch.nan_to_num(
        base_pred_audio.to(dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    batch_size, num_stems = safe_audio.shape[:2]
    threshold = dbfs_to_amplitude(threshold_db)
    window_samples = max(1, int(round(sample_rate * max(float(window_ms), 1.0) / 1000.0)))
    if window_samples % 2 == 0:
        window_samples += 1

    power = safe_audio.pow(2).mean(dim=2)
    power = rearrange(power, "b n t -> (b n) 1 t")
    if window_samples > 1:
        power = F.avg_pool1d(
            power,
            kernel_size=window_samples,
            stride=1,
            padding=window_samples // 2,
        )
    rms = power.clamp_min(0.0).sqrt()
    active = rms > threshold
    min_active_samples = max(0, int(round(sample_rate * max(float(min_active_ms), 0.0) / 1000.0)))
    active = _remove_short_active_runs(active, min_active_samples)
    gap_fill_samples = max(0, int(round(sample_rate * max(float(gap_fill_ms), 0.0) / 1000.0)))
    active = _fill_short_inactive_gaps(active, gap_fill_samples)
    protect_ms_value = max(float(protect_ms), 0.0)
    pre_ms_value = protect_ms_value if pre_protect_ms is None else max(float(pre_protect_ms), 0.0)
    post_ms_value = protect_ms_value if post_protect_ms is None else max(float(post_protect_ms), 0.0)
    pre_protect_samples = max(0, int(round(sample_rate * pre_ms_value / 1000.0)))
    post_protect_samples = max(0, int(round(sample_rate * post_ms_value / 1000.0)))
    active = _dilate_active_regions(
        active,
        pre_protect_samples=pre_protect_samples,
        post_protect_samples=post_protect_samples,
    )
    return rearrange(
        active,
        "(b n) 1 t -> b n 1 t",
        b=batch_size,
        n=num_stems,
    )


def build_allocator_local_activity_gate(
    base_pred_audio: torch.Tensor,
    *,
    sample_rate: int,
    threshold_db: float,
    window_ms: float,
    protect_ms: float = 0.0,
    pre_protect_ms: float | None = None,
    post_protect_ms: float | None = None,
    min_active_ms: float = 0.0,
    gap_fill_ms: float = 0.0,
    transition_db: float = 6.0,
    protect_floor: float = 0.25,
) -> torch.Tensor:
    safe_audio = torch.nan_to_num(
        base_pred_audio.to(dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    batch_size, num_stems = safe_audio.shape[:2]
    threshold_db = min(float(threshold_db), 0.0)
    transition_db = max(float(transition_db), 1e-3)
    protect_floor = min(max(float(protect_floor), 0.0), 1.0)
    window_samples = max(1, int(round(sample_rate * max(float(window_ms), 1.0) / 1000.0)))
    if window_samples % 2 == 0:
        window_samples += 1

    power = safe_audio.pow(2).mean(dim=2)
    power = rearrange(power, "b n t -> (b n) 1 t")
    if window_samples > 1:
        power = F.avg_pool1d(
            power,
            kernel_size=window_samples,
            stride=1,
            padding=window_samples // 2,
        )
    rms = power.clamp_min(0.0).sqrt()
    rms_db = 20.0 * torch.log10(rms.clamp_min(1e-8))
    lower_db = threshold_db - transition_db
    keep_gate = ((rms_db - lower_db) / transition_db).clamp(0.0, 1.0)
    active = rms_db > threshold_db
    min_active_samples = max(0, int(round(sample_rate * max(float(min_active_ms), 0.0) / 1000.0)))
    active = _remove_short_active_runs(active, min_active_samples)
    gap_fill_samples = max(0, int(round(sample_rate * max(float(gap_fill_ms), 0.0) / 1000.0)))
    active = _fill_short_inactive_gaps(active, gap_fill_samples)
    protect_ms_value = max(float(protect_ms), 0.0)
    pre_ms_value = protect_ms_value if pre_protect_ms is None else max(float(pre_protect_ms), 0.0)
    post_ms_value = protect_ms_value if post_protect_ms is None else max(float(post_protect_ms), 0.0)
    pre_protect_samples = max(0, int(round(sample_rate * pre_ms_value / 1000.0)))
    post_protect_samples = max(0, int(round(sample_rate * post_ms_value / 1000.0)))
    protected_active = _dilate_active_regions(
        active,
        pre_protect_samples=pre_protect_samples,
        post_protect_samples=post_protect_samples,
    )
    keep_gate = torch.maximum(keep_gate, active.to(dtype=keep_gate.dtype))
    protected_only = protected_active & ~active
    if protect_floor > 0.0:
        keep_gate = torch.maximum(
            keep_gate,
            protected_only.to(dtype=keep_gate.dtype) * protect_floor,
        )

    return rearrange(
        keep_gate,
        "(b n) 1 t -> b n 1 t",
        b=batch_size,
        n=num_stems,
    )


def is_vocal_stem_name(stem_name: str) -> bool:
    lowered = str(stem_name or "").strip().lower()
    return ("vocal" in lowered) or ("vox" in lowered)


def is_bass_stem_name(stem_name: str) -> bool:
    lowered = str(stem_name or "").strip().lower()
    return lowered == "bass" or lowered.endswith("_bass") or "bass" in lowered


def fft_highpass_waveform(
    audio: torch.Tensor,
    *,
    sample_rate: int,
    cutoff_hz: float,
    transition_hz: float,
) -> torch.Tensor:
    cutoff = max(float(cutoff_hz), 0.0)
    transition = max(float(transition_hz), 1.0)
    num_samples = int(audio.shape[-1])
    if num_samples <= 1 or cutoff >= (float(sample_rate) * 0.5):
        return torch.zeros_like(audio)

    spec = torch.fft.rfft(audio.float(), dim=-1)
    freqs = torch.fft.rfftfreq(
        num_samples,
        d=1.0 / float(sample_rate),
        device=audio.device,
    ).to(dtype=audio.dtype)
    mask = ((freqs - cutoff) / transition).clamp(0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)
    high = torch.fft.irfft(spec * mask.to(dtype=spec.dtype), n=num_samples, dim=-1)
    return high.to(dtype=audio.dtype)


def local_rms_db(
    audio: torch.Tensor,
    *,
    sample_rate: int,
    window_ms: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    if audio.ndim != 3:
        raise ValueError(f"Expected audio as B,C,T, got {audio.shape}")
    target_length = int(audio.shape[-1])
    window_samples = max(1, int(round(float(sample_rate) * max(float(window_ms), 1.0) / 1000.0)))
    power = audio.float().pow(2).mean(dim=1, keepdim=True)
    if window_samples > 1:
        power = F.avg_pool1d(
            power,
            kernel_size=window_samples,
            stride=1,
            padding=window_samples // 2,
        )
        if power.shape[-1] > target_length:
            power = power[..., :target_length]
        elif power.shape[-1] < target_length:
            power = F.pad(power, (0, target_length - power.shape[-1]), mode="replicate")
    return 10.0 * torch.log10(power.clamp_min(eps))


def apply_bass_high_closure_guard(
    refined: torch.Tensor,
    pre_closure: torch.Tensor,
    base_pred: torch.Tensor,
    ordered_instruments: list[str],
    *,
    sample_rate: int,
    cutoff_hz: float,
    transition_hz: float,
    protect_threshold_db: float,
    max_residual_db: float,
    window_ms: float,
    gate_transition_db: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bass_indices = [idx for idx, name in enumerate(ordered_instruments) if is_bass_stem_name(name)]
    if not bass_indices:
        zeros = torch.zeros_like(refined)
        return refined, zeros, zeros

    bass_idx = bass_indices[0]
    recipient_indices = [idx for idx in range(len(ordered_instruments)) if idx != bass_idx]
    if not recipient_indices:
        zeros = torch.zeros_like(refined)
        return refined, zeros, zeros

    closure_delta = refined[:, bass_idx] - pre_closure[:, bass_idx]
    closure_high = fft_highpass_waveform(
        closure_delta,
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        transition_hz=transition_hz,
    )
    base_high = fft_highpass_waveform(
        base_pred[:, bass_idx],
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        transition_hz=transition_hz,
    )
    pre_high = fft_highpass_waveform(
        pre_closure[:, bass_idx],
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        transition_hz=transition_hz,
    )

    protect_db = torch.maximum(
        local_rms_db(base_high, sample_rate=sample_rate, window_ms=window_ms),
        local_rms_db(pre_high, sample_rate=sample_rate, window_ms=window_ms),
    )
    closure_db = local_rms_db(closure_high, sample_rate=sample_rate, window_ms=window_ms)
    transition_db = max(float(gate_transition_db), 0.1)
    protect_gate = ((float(protect_threshold_db) - protect_db) / transition_db).clamp(0.0, 1.0)
    residual_gate = ((float(max_residual_db) - closure_db) / transition_db).clamp(0.0, 1.0)
    guard_gate = (protect_gate * residual_gate).to(dtype=refined.dtype)
    removed = closure_high * guard_gate

    if removed.abs().amax() <= 0:
        zeros = torch.zeros_like(refined)
        return refined, zeros, zeros

    recipient_ref = pre_closure[:, recipient_indices]
    recipient_high = fft_highpass_waveform(
        recipient_ref,
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        transition_hz=transition_hz,
    )
    energy = recipient_high.float().pow(2).mean(dim=2, keepdim=True)
    window_samples = max(1, int(round(float(sample_rate) * max(float(window_ms), 1.0) / 1000.0)))
    if window_samples > 1:
        flat_energy = rearrange(energy, "b n c t -> (b n) c t")
        flat_energy = F.avg_pool1d(
            flat_energy,
            kernel_size=window_samples,
            stride=1,
            padding=window_samples // 2,
        )
        if flat_energy.shape[-1] > refined.shape[-1]:
            flat_energy = flat_energy[..., : refined.shape[-1]]
        elif flat_energy.shape[-1] < refined.shape[-1]:
            flat_energy = F.pad(flat_energy, (0, refined.shape[-1] - flat_energy.shape[-1]), mode="replicate")
        energy = rearrange(flat_energy, "(b n) c t -> b n c t", b=refined.shape[0], n=len(recipient_indices))

    weights_sum = energy.sum(dim=1, keepdim=True)
    weights = energy / weights_sum.clamp_min(1e-12)
    if "other" in [name.lower() for name in ordered_instruments]:
        fallback = torch.zeros_like(weights)
        other_idx = next(
            (out_idx for out_idx, stem_idx in enumerate(recipient_indices) if ordered_instruments[stem_idx].lower() == "other"),
            0,
        )
        fallback[:, other_idx] = 1.0
    else:
        fallback = torch.full_like(weights, 1.0 / float(len(recipient_indices)))
    weights = torch.where(weights_sum > 1e-12, weights, fallback)

    redistribution = weights.to(dtype=refined.dtype) * rearrange(removed, "b c t -> b 1 c t")
    guarded = refined.clone()
    guarded[:, bass_idx] = guarded[:, bass_idx] - removed
    guarded[:, recipient_indices] = guarded[:, recipient_indices] + redistribution

    removed_by_stem = torch.zeros_like(refined)
    redistributed_by_stem = torch.zeros_like(refined)
    removed_by_stem[:, bass_idx] = removed
    redistributed_by_stem[:, recipient_indices] = redistribution
    return guarded, removed_by_stem, redistributed_by_stem


def build_source_recipient_allow_matrix(
    ordered_instruments: list[str],
    *,
    device: torch.device,
) -> torch.Tensor:
    num_stems = len(ordered_instruments)
    allow = torch.ones(num_stems, num_stems, dtype=torch.bool, device=device)
    vocal_indices = [idx for idx, name in enumerate(ordered_instruments) if is_vocal_stem_name(name)]
    if not vocal_indices or len(vocal_indices) >= num_stems:
        return allow

    for source_idx, name in enumerate(ordered_instruments):
        if not is_vocal_stem_name(name):
            allow[source_idx, vocal_indices] = False
    return allow


def resize_allocator_waveform_weights(
    weights: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    if weights.ndim == 3:
        batch_size, num_stems = weights.shape[:2]
        flat = rearrange(weights, "b n t -> (b n) 1 t")
        num_channels = 1
    elif weights.ndim == 4:
        batch_size, num_stems, num_channels = weights.shape[:3]
        flat = rearrange(weights, "b n c t -> (b n) c t")
    else:
        raise ValueError(f"Unsupported waveform weight rank: {weights.shape}")

    if flat.shape[-1] != target_length:
        flat = F.interpolate(flat, size=target_length, mode="nearest")

    return rearrange(flat, "(b n) c t -> b n c t", b=batch_size, n=num_stems, c=num_channels)


def extract_allocator_waveform_routing_weights(
    aux: dict[str, torch.Tensor],
    *,
    target_length: int,
) -> torch.Tensor | None:
    closure_weights = aux.get("closure_weights", None)
    if isinstance(closure_weights, torch.Tensor):
        return resize_allocator_waveform_weights(closure_weights.float(), target_length)

    routing_weights = aux.get("weights", None)
    if isinstance(routing_weights, torch.Tensor):
        routing_weights = routing_weights.float().mean(dim=2, keepdim=True)
        return resize_allocator_waveform_weights(routing_weights, target_length)
    return None


def redistribute_removed_base_sections(
    removed_sections: torch.Tensor,
    *,
    ordered_instruments: list[str],
    aux: dict[str, torch.Tensor],
) -> torch.Tensor:
    eps = 1e-8
    if removed_sections.numel() == 0:
        return removed_sections

    allow_matrix = build_source_recipient_allow_matrix(
        ordered_instruments,
        device=removed_sections.device,
    )
    routing_weights = extract_allocator_waveform_routing_weights(
        aux,
        target_length=removed_sections.shape[-1],
    )

    redistributed = torch.zeros_like(removed_sections)
    batch_size, num_stems, num_channels, target_length = removed_sections.shape

    for source_idx in range(num_stems):
        source_removed = removed_sections[:, source_idx : source_idx + 1]
        if not torch.any(source_removed.abs() > eps):
            continue

        allowed = allow_matrix[source_idx].view(1, num_stems, 1, 1)
        fallback = allowed.to(dtype=removed_sections.dtype).expand(batch_size, -1, num_channels, target_length)
        fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(eps)

        if routing_weights is None:
            source_weights = fallback
        else:
            masked_weights = routing_weights.to(dtype=removed_sections.dtype) * allowed.to(
                dtype=removed_sections.dtype
            )
            if masked_weights.shape[2] == 1 and num_channels != 1:
                masked_weights = masked_weights.expand(-1, -1, num_channels, -1)
            masked_sum = masked_weights.sum(dim=1, keepdim=True)
            source_weights = torch.where(
                masked_sum > eps,
                masked_weights / masked_sum.clamp_min(eps),
                fallback,
            )

        redistributed = redistributed + source_weights * source_removed

    return redistributed


def maybe_prune_base_silent_stems_for_allocator(
    waveforms: dict[str, np.ndarray],
    ordered_instruments: list[str],
    *,
    sample_rate: int,
    prune_base_silent_stems: bool,
    base_silent_threshold_db: float,
    base_silent_window_ms: float,
) -> tuple[dict[str, np.ndarray], list[str]]:
    if not prune_base_silent_stems:
        return waveforms, []

    threshold = dbfs_to_amplitude(base_silent_threshold_db)
    window_samples = max(1, int(round(sample_rate * max(float(base_silent_window_ms), 1.0) / 1000.0)))
    pruned: list[str] = []
    updated = dict(waveforms)

    for instr in ordered_instruments:
        activity = waveform_max_rms(updated[instr], window_samples)
        if activity <= threshold:
            updated[instr] = np.zeros_like(updated[instr], dtype=np.float32)
            pruned.append(instr)

    if pruned:
        print(
            "Pruning base-silent stems before allocator "
            f"(RMS <= {min(float(base_silent_threshold_db), 0.0):.1f} dBFS / {float(base_silent_window_ms):.1f} ms): "
            + ", ".join(pruned)
        )
    return updated, pruned


def maybe_filter_silent_stems(
    waveforms: dict[str, np.ndarray],
    ordered_instruments: list[str],
    *,
    skip_silent_stems: bool,
    silent_stem_threshold_db: float,
) -> tuple[list[str], list[str]]:
    ordered = list(ordered_instruments)
    if not skip_silent_stems:
        return ordered, []

    peak_threshold = dbfs_to_amplitude(silent_stem_threshold_db)
    kept: list[str] = []
    skipped: list[str] = []
    for instr in ordered:
        peak = waveform_peak(waveforms[instr])
        if peak <= peak_threshold:
            skipped.append(instr)
        else:
            kept.append(instr)

    if skipped:
        threshold_label = f"{min(float(silent_stem_threshold_db), 0.0):.1f} dBFS"
        print(f"Skipping silent stems (peak <= {threshold_label}): {', '.join(skipped)}")
        if not kept:
            print("All final stems met the silence threshold; nothing will be written.")
    return kept, skipped


def _format_db(value: float) -> str:
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.2f}"


def compute_mix_closure_metrics(
    *,
    waveforms: dict[str, np.ndarray],
    ordered_instruments: list[str],
    mix_orig: np.ndarray,
    config: ConfigDict,
    norm_params: dict[str, float] | None,
) -> dict[str, Any]:
    closure_stems = [instr for instr in ordered_instruments if instr in waveforms]
    if not closure_stems:
        return {
            "percent": float("nan"),
            "residual_rms_percent": float("nan"),
            "residual_rms_db": float("nan"),
            "residual_max_abs": float("nan"),
            "residual_peak": float("nan"),
            "mix_rms": float("nan"),
            "residual_rms": float("nan"),
            "stems": [],
        }

    closure_waveforms = {
        instr: np.asarray(waveforms[instr], dtype=np.float64)
        for instr in closure_stems
    }
    if norm_params is not None and bool(get_inference_setting(config, "normalize", False)):
        closure_waveforms = {
            instr: denormalize_audio(audio, norm_params).astype(np.float64, copy=False)
            for instr, audio in closure_waveforms.items()
        }

    mix_ref = np.asarray(mix_orig, dtype=np.float64)
    min_channels = min([mix_ref.shape[0], *[audio.shape[0] for audio in closure_waveforms.values()]])
    min_length = min([mix_ref.shape[-1], *[audio.shape[-1] for audio in closure_waveforms.values()]])
    mix_ref = mix_ref[:min_channels, :min_length]
    stem_sum = np.zeros_like(mix_ref, dtype=np.float64)
    for instr in closure_stems:
        stem_sum += closure_waveforms[instr][:min_channels, :min_length]

    residual = mix_ref - stem_sum
    mix_rms = float(np.sqrt(np.mean(np.square(mix_ref))) + 1e-12)
    residual_rms = float(np.sqrt(np.mean(np.square(residual))) + 1e-12)
    residual_peak = float(np.max(np.abs(residual))) if residual.size else 0.0
    if mix_rms <= 1e-12:
        residual_ratio = 0.0 if residual_rms <= 1e-12 else float("inf")
    else:
        residual_ratio = residual_rms / mix_rms
    closure_percent = 100.0 * (1.0 - residual_ratio) if math.isfinite(residual_ratio) else float("-inf")
    residual_rms_db = 20.0 * math.log10(residual_ratio) if residual_ratio > 0.0 and math.isfinite(residual_ratio) else (
        float("-inf") if residual_ratio == 0.0 else float("inf")
    )
    return {
        "percent": float(closure_percent),
        "residual_rms_percent": float(100.0 * residual_ratio) if math.isfinite(residual_ratio) else float("inf"),
        "residual_rms_db": float(residual_rms_db),
        "residual_max_abs": residual_peak,
        "residual_peak": residual_peak,
        "mix_rms": mix_rms,
        "residual_rms": residual_rms,
        "stems": list(closure_stems),
    }


def print_mix_closure_metrics(metrics: dict[str, Any]) -> None:
    percent = float(metrics.get("percent", float("nan")))
    residual_percent = float(metrics.get("residual_rms_percent", float("nan")))
    residual_db = float(metrics.get("residual_rms_db", float("nan")))
    residual_max_abs = float(
        metrics.get("residual_max_abs", metrics.get("residual_peak", float("nan")))
    )
    stems = ", ".join(str(stem) for stem in metrics.get("stems", []))
    print(
        "Mix closure: "
        f"{percent:.6f}% "
        f"(residual RMS {residual_percent:.6f}% of mix, {_format_db(residual_db)} dB, "
        f"residual max_abs {residual_max_abs:.8f}; stems: {stems})"
    )


def save_estimates(
    *,
    waveforms: dict[str, np.ndarray],
    ordered_instruments: list[str],
    mix_orig: np.ndarray,
    norm_params: dict[str, float] | None,
    args: argparse.Namespace,
    config: ConfigDict,
    sample_rate: int,
    force_skip_instruments: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    output_root = get_output_root(args)

    if norm_params is not None and bool(get_inference_setting(config, "normalize", False)):
        waveforms = {
            instr: denormalize_audio(estimates, norm_params)
            for instr, estimates in waveforms.items()
        }

    waveforms, ordered_instruments = maybe_add_derived_instrumentals(
        waveforms,
        mix_orig,
        ordered_instruments,
        extract_instrumental=args.extract_instrumental,
        save_instrumental_variants=args.save_instrumental_variants,
        save_mix_minus_instrumental_sum_non_vocals=args.save_mixture_minus_instrumental_sum_non_vocals,
    )
    waveforms, ordered_instruments = maybe_add_collapsed_four_stem_from_six(
        waveforms,
        mix_orig,
        ordered_instruments,
        collapse_to_4stem=args.collapse_to_4stem,
    )
    waveforms, ordered_instruments = maybe_project_to_four_stem_plus_instrumental(
        waveforms,
        mix_orig,
        ordered_instruments,
        output_4stem_plus_instrumental=args.output_4stem_plus_instrumental,
    )
    forced_skipped: list[str] = []
    if force_skip_instruments:
        force_skip_set = set(force_skip_instruments)
        forced_skipped = [instr for instr in ordered_instruments if instr in force_skip_set]
        if forced_skipped:
            ordered_instruments = [instr for instr in ordered_instruments if instr not in force_skip_set]
            print("Omitting base-silent stems from write-out: " + ", ".join(forced_skipped))
    ordered_instruments, skipped_instruments = maybe_filter_silent_stems(
        waveforms,
        ordered_instruments,
        skip_silent_stems=args.skip_silent_stems,
        silent_stem_threshold_db=args.silent_stem_threshold_db,
    )
    skipped_instruments = [*forced_skipped, *skipped_instruments]

    for instr in ordered_instruments:
        save_audio_file(output_root / instr, waveforms[instr], sample_rate, args)
    return ordered_instruments, skipped_instruments


def save_refiner_diagnostics(
    diagnostics: dict[str, dict[str, np.ndarray] | np.ndarray],
    *,
    ordered_instruments: list[str],
    args: argparse.Namespace,
    config: ConfigDict,
    sample_rate: int,
    norm_params: dict[str, float] | None,
) -> None:
    base_dir = get_output_root(args) / "refiner"
    base_dir.mkdir(parents=True, exist_ok=True)
    control_categories = {
        "stem_reclaim_keep_mask",
        "stem_delta_gate",
        "gate_confidence",
        "stem_activity_mask",
    }

    for category in (
        "routed_residual",
        "direct_delta",
        "total_delta",
        "closure_delta",
        "bass_high_closure_guard_removed",
        "bass_high_closure_guard_redistributed",
        "base_low_removed_source",
        "base_low_redistributed",
        "stem_reclaim_keep_mask",
        "stem_delta_gate",
        "gate_confidence",
        "stem_activity_mask",
    ):
        category_waveforms = diagnostics.get(category, None)
        if not isinstance(category_waveforms, dict):
            continue
        category_dir = base_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for instr in ordered_instruments:
            if category in control_categories:
                estimates = np.clip(
                    np.asarray(category_waveforms[instr], dtype=np.float32),
                    a_min=0.0,
                    a_max=1.0,
                )
            else:
                estimates = scale_difference_waveform_if_needed(
                    category_waveforms[instr],
                    config,
                    norm_params,
                )
            save_audio_file(category_dir / instr, estimates, sample_rate, args)
            if category not in control_categories:
                sum_part, sub_part = split_signed_waveform(estimates)
                save_audio_file(category_dir / f"sum_{instr}", sum_part, sample_rate, args)
                save_audio_file(category_dir / f"sub_{instr}", sub_part, sample_rate, args)

    remaining_residual = diagnostics.get("remaining_residual", None)
    if isinstance(remaining_residual, np.ndarray):
        remaining_residual = scale_difference_waveform_if_needed(
            remaining_residual,
            config,
            norm_params,
        )
        save_audio_file(base_dir / "remaining_residual", remaining_residual, sample_rate, args)
        sum_part, sub_part = split_signed_waveform(remaining_residual)
        save_audio_file(base_dir / "sum_remaining_residual", sum_part, sample_rate, args)
        save_audio_file(base_dir / "sub_remaining_residual", sub_part, sample_rate, args)


def emit_progress(progress_callback: ProgressCallback | None, stage: str, value: int) -> None:
    if progress_callback is None:
        return
    progress_callback(stage, value)


def load_inference_runtime(
    args: argparse.Namespace,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    ensure_exists(args.base_config, "Base config")
    ensure_exists(args.base_checkpoint, "Base checkpoint")
    if not args.base_only:
        ensure_exists(args.allocator_checkpoint, "Allocator checkpoint")
    emit_progress(progress_callback, "validating", 4)

    device = resolve_device(args.device, args.cuda_device)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    print(f"Using device: {device}")
    print(f"Inference mode: {'base-only' if args.base_only else 'allocator-refine'}")
    print(f"Base config: {args.base_config}")
    print(f"Base checkpoint: {args.base_checkpoint}")
    if args.base_only:
        if args.save_diagnostics:
            print("Ignoring --save-diagnostics because --base-only was requested.")
    else:
        print(f"Allocator checkpoint: {args.allocator_checkpoint}")
        if args.allocator_config.is_file():
            print(f"Allocator config: {args.allocator_config}")
        else:
            print("Allocator config: using legacy run_config from checkpoint")

    emit_progress(progress_callback, "loading_models", 12)
    model_load_started = time.time()
    base_model, base_config = get_model_from_config(args.base_config)
    base_checkpoint = load_checkpoint(args.base_checkpoint, weights_only=False, map_location="cpu")
    load_model_state(base_model, base_checkpoint, args.base_checkpoint)
    base_model = base_model.to(device)
    base_model.eval()

    allocator = None
    if not args.base_only:
        allocator_checkpoint = load_checkpoint(
            args.allocator_checkpoint,
            weights_only=False,
            map_location="cpu",
        )
        run_config = load_allocator_run_config(
            allocator_config_path=args.allocator_config,
            allocator_checkpoint=allocator_checkpoint,
        )
        allocator_state_dict = normalize_checkpoint_state_dict(allocator_checkpoint)
        if not allocator_state_dict:
            raise ValueError(f"Allocator checkpoint has no tensors: {args.allocator_checkpoint}")
        allocator = build_allocator_from_run_config(
            run_config,
            allocator_state_dict,
            base_config,
            device,
        )
        if hasattr(allocator, "inactive_conf_scale"):
            allocator.inactive_conf_scale = min(
                max(float(args.allocator_base_low_section_inactive_conf_scale), 0.0),
                1.0,
            )
        if hasattr(allocator, "inactive_keep_floor"):
            allocator.inactive_keep_floor = min(
                max(float(args.allocator_base_low_section_protect_floor), 0.0),
                1.0,
            )
        if hasattr(allocator, "inactive_keep_max"):
            allocator.inactive_keep_max = min(
                max(float(args.allocator_base_low_section_inactive_keep_max), 0.0),
                1.0,
            )

    print(f"Model load time: {time.time() - model_load_started:.2f} sec")
    print(f"Instruments: {list(base_config.training.instruments)}")
    if not args.base_only:
        allocator_chunk_size = int(
            get_inference_setting(
                base_config,
                "allocator_chunk_size",
                get_inference_setting(base_config, "chunk_size", base_config.audio.chunk_size),
            )
        )
        print(f"Allocator chunk size: {allocator_chunk_size}")
    stem_chunk_overrides = get_stem_chunk_size_overrides(base_config)
    if stem_chunk_overrides:
        print(f"Stem chunk overrides: {stem_chunk_overrides}")

    return {
        "device": device,
        "base_model": base_model,
        "base_config": base_config,
        "allocator": allocator,
    }


def run_single_inference(
    args: argparse.Namespace,
    progress_callback: ProgressCallback | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_exists(args.input, "Input audio")
    if runtime is None:
        runtime = load_inference_runtime(args, progress_callback)

    device = runtime["device"]
    base_model = runtime["base_model"]
    base_config = runtime["base_config"]
    allocator = runtime["allocator"]

    emit_progress(progress_callback, "loading_audio", 28)
    sample_rate = int(getattr(base_config.audio, "sample_rate", 44100))
    expected_channels = getattr(base_config.audio, "num_channels", None)
    mix, sr = load_audio(args.input, sample_rate, expected_channels)
    mix_orig = mix.copy()

    norm_params = None
    if bool(get_inference_setting(base_config, "normalize", False)):
        mix, norm_params = normalize_audio(mix)

    start_time = time.time()
    show_progress = not args.disable_progress

    emit_progress(progress_callback, "demixing", 46)
    waveforms = demix(
        config=base_config,
        model=base_model,
        mix=mix,
        device=device,
        show_progress=show_progress,
    )
    if args.use_tta:
        emit_progress(progress_callback, "applying_tta", 62)
        waveforms = apply_tta(
            config=base_config,
            model=base_model,
            mix=mix,
            waveforms_orig=waveforms,
            device=device,
            show_progress=show_progress,
        )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.base_only:
        refined_waveforms = dict(waveforms)
        diagnostic_outputs = None
        pruned_base_silent_stems: list[str] = []
        refined_waveforms, ordered_instruments = maybe_add_complementary_stem(
            refined_waveforms,
            mix,
            base_config,
        )
    else:
        emit_progress(progress_callback, "refining", 78)
        refined_waveforms, ordered_instruments, diagnostic_outputs, pruned_base_silent_stems = refine_with_allocator_chunked(
            allocator=allocator,
            base_model=base_model,
            base_config=base_config,
            mix=mix,
            base_waveforms=waveforms,
            device=device,
            show_progress=show_progress,
            collect_diagnostics=args.save_diagnostics,
            prune_base_silent_stems=args.allocator_prune_base_silent_stems,
            base_silent_threshold_db=args.allocator_base_silent_threshold_db,
            base_silent_window_ms=args.allocator_base_silent_window_ms,
            residualize_base_low_sections=args.allocator_residualize_base_low_sections,
            base_low_section_threshold_db=args.allocator_base_low_section_threshold_db,
            base_low_section_window_ms=args.allocator_base_low_section_window_ms,
            base_low_section_fade_protect_ms=args.allocator_base_low_section_fade_protect_ms,
            base_low_section_pre_protect_ms=args.allocator_base_low_section_pre_protect_ms,
            base_low_section_post_protect_ms=args.allocator_base_low_section_post_protect_ms,
            base_low_section_transition_db=args.allocator_base_low_section_transition_db,
            base_low_section_min_active_ms=args.allocator_base_low_section_min_active_ms,
            base_low_section_gap_fill_ms=args.allocator_base_low_section_gap_fill_ms,
            base_low_section_protect_floor=args.allocator_base_low_section_protect_floor,
            mask_silent_sections=args.allocator_mask_silent_sections,
            mask_silent_threshold_db=args.allocator_mask_silent_threshold_db,
            mask_silent_window_ms=args.allocator_mask_silent_window_ms,
            guard_bass_high_closure=args.allocator_guard_bass_high_closure,
            bass_high_closure_cutoff_hz=args.allocator_bass_high_closure_cutoff_hz,
            bass_high_closure_transition_hz=args.allocator_bass_high_closure_transition_hz,
            bass_high_closure_protect_threshold_db=args.allocator_bass_high_closure_protect_threshold_db,
            bass_high_closure_max_residual_db=args.allocator_bass_high_closure_max_residual_db,
            bass_high_closure_window_ms=args.allocator_bass_high_closure_window_ms,
            bass_high_closure_gate_transition_db=args.allocator_bass_high_closure_gate_transition_db,
        )
        refined_waveforms, ordered_instruments = maybe_add_complementary_stem(
            refined_waveforms,
            mix,
            base_config,
        )

    closure_metrics = compute_mix_closure_metrics(
        waveforms=refined_waveforms,
        ordered_instruments=ordered_instruments,
        mix_orig=mix_orig,
        config=base_config,
        norm_params=norm_params,
    )
    print_mix_closure_metrics(closure_metrics)

    emit_progress(progress_callback, "saving", 92)
    output_root = get_output_root(args)
    output_root.mkdir(parents=True, exist_ok=True)
    saved_instruments, skipped_silent_stems = save_estimates(
        waveforms=refined_waveforms,
        ordered_instruments=ordered_instruments,
        mix_orig=mix_orig,
        norm_params=norm_params,
        args=args,
        config=base_config,
        sample_rate=sr,
        force_skip_instruments=pruned_base_silent_stems,
    )

    if diagnostic_outputs is not None:
        save_refiner_diagnostics(
            diagnostic_outputs,
            ordered_instruments=saved_instruments,
            args=args,
            config=base_config,
            sample_rate=sr,
            norm_params=norm_params,
        )

    elapsed = time.time() - start_time
    emit_progress(progress_callback, "complete", 100)
    print(f"Stem output directory: {output_root}")
    print(f"Elapsed time: {elapsed:.2f} sec")
    return {
        "device": str(device),
        "mode": "base-only" if args.base_only else "allocator-refine",
        "output_root": str(output_root),
        "sample_rate": sr,
        "elapsed_time_sec": elapsed,
        "diagnostics_saved": diagnostic_outputs is not None,
        "ordered_instruments": list(saved_instruments),
        "skipped_silent_stems": list(skipped_silent_stems),
        "pruned_base_silent_stems": list(pruned_base_silent_stems),
        "mix_closure": closure_metrics,
    }


def run_folder_inference(
    args: argparse.Namespace,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    ensure_input_exists(args.input)
    extensions = parse_input_extensions(args.input_extensions)
    input_files = iter_input_files(
        args.input,
        recursive=bool(args.recursive),
        extensions=extensions,
    )
    if not input_files:
        extensions_label = ", ".join(sorted(extensions))
        raise FileNotFoundError(f"No audio files found in {args.input} with extensions: {extensions_label}")
    if args.output_dir_is_stem_dir and len(input_files) > 1:
        raise ValueError("--output-dir-is-stem-dir can only be used with a single input file.")

    runtime = load_inference_runtime(args, progress_callback)
    results: list[dict[str, Any]] = []
    input_root = args.input.resolve()
    total = len(input_files)
    for index, input_file in enumerate(input_files, start=1):
        track_args = argparse.Namespace(**vars(args))
        track_args.input = input_file
        if args.input.is_dir() and not args.output_dir_is_stem_dir:
            track_args._batch_relative_stem = get_batch_relative_stem(input_root, input_file)

        print(f"\n[{index}/{total}] Input: {input_file}")
        print(f"[{index}/{total}] Stem output directory: {get_output_root(track_args)}")
        results.append(
            run_single_inference(
                track_args,
                progress_callback=progress_callback,
                runtime=runtime,
            )
        )

    return {
        "mode": "folder",
        "input_root": str(input_root),
        "num_inputs": total,
        "results": results,
    }


def run_inference(
    args: argparse.Namespace,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    ensure_input_exists(args.input)
    if args.input.is_dir():
        return run_folder_inference(args, progress_callback)
    return run_single_inference(args, progress_callback)


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
