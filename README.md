# Residual Allocator

Residual Allocator is a lightweight post-refinement module for BS-RoFormer stem
separation. It routes the residual between the input mix and base separator
outputs back into the most plausible stems.

Conventional stem separators can produce clean stems while still leaving a
non-trivial gap between the input mix and the sum of separated stems. Residual
Allocator treats that reconstruction residual as musically meaningful missing
content, then routes it back into plausible stems while preserving mix closure.

This repository contains the inference code, model definition, base BS-RoFormer
architecture code needed for loading checkpoints, and a ConvResidualAllocator
inference checkpoint.

## What This Is

Core:

- Residual-aware post-refinement for compatible BS-RoFormer outputs.
- Standalone inference for single files and folders.
- A small allocator checkpoint distributed as `safetensors` plus sidecar YAML.

Not core:

- This is not a standalone separator. It runs after a compatible BS-RoFormer
  base checkpoint.
- The stem upmix script under `examples/` is an application example, not the
  main allocator method.

## Checkpoints

The allocator is distributed as:

- `weights/residual_allocator.safetensors`
- `configs/residual_allocator.yaml`

The original training `.ckpt` format is intentionally not required for inference.
The sidecar YAML keeps architecture/configuration separate from tensor weights,
which makes the public artifact easier to inspect and safer to load.

The base BS-RoFormer checkpoint is not included here. Download a compatible
checkpoint from
[jarredou/BS-ROFO-SW-Fixed](https://huggingface.co/jarredou/BS-ROFO-SW-Fixed)
and place it at `weights/BS-Rofo-SW-Fixed.ckpt`, or pass it explicitly with
`--base-checkpoint`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inference

Single-file inference:

```bash
python infer.py \
  --input /path/to/song.wav \
  --output outputs/song \
  --output-dir-is-stem-dir \
  --base-checkpoint weights/BS-Rofo-SW-Fixed.ckpt
```

Folder inference:

```bash
python infer.py \
  --input /path/to/audio_folder \
  --output outputs \
  --recursive \
  --base-checkpoint weights/BS-Rofo-SW-Fixed.ckpt
```

For folder inference, each input file is written to its own stem directory under
`--output`. For example, `/path/to/audio_folder/album/song.wav` becomes
`outputs/album/song/` when `--recursive` is used.

Useful options:

- `--output-dir-is-stem-dir`: write stems directly into `--output`.
- `--recursive`: search an input directory recursively.
- `--input-extensions`: comma-separated extensions used for folder inference.
- `--base-only`: run the base BS-RoFormer without allocator refinement.
- `--allocator-checkpoint`: path to allocator `.safetensors` or legacy `.ckpt`.
- `--allocator-config`: path to allocator sidecar YAML.
- `--extract-instrumental`: also save `instrumental = mix - vocals`.

## Stem Layouts

Residual Allocator follows the stem layout returned by the base BS-RoFormer
checkpoint. There is no separate allocator mode switch for 4-stem versus
6-stem inference.

- With a 6-stem base model, residuals are routed across `bass`, `drums`,
  `other`, `vocals`, `guitar`, and `piano`.
- With a 4-stem base model, residuals are routed across `bass`, `drums`,
  `other`, and `vocals`. Material that has no separate guitar or piano slot is
  naturally handled through the base model's `other` stem.

If you use a 6-stem base model but want 4-stem deliverables, use
`--output-4stem-plus-instrumental`.

## Compare Against Base

Run the same track once with the base separator and once with Residual
Allocator. The CLI prints mix closure metrics, including residual RMS and
residual max amplitude, so the sum of stems can be compared against the input
mix.

```bash
python infer.py \
  --input /path/to/song.wav \
  --output outputs/base/song \
  --output-dir-is-stem-dir \
  --base-only \
  --base-checkpoint weights/BS-Rofo-SW-Fixed.ckpt

python infer.py \
  --input /path/to/song.wav \
  --output outputs/allocator/song \
  --output-dir-is-stem-dir \
  --base-checkpoint weights/BS-Rofo-SW-Fixed.ckpt
```

Useful comparisons:

- Base BS-RoFormer stems.
- Base + Residual Allocator stems.
- Sum of stems versus the original input mix.
- Stem-level listening checks for residual material, artifacts, and leakage.

Example closure metrics on a private full-length stereo reference track
(280.6 seconds, not redistributed):

| mode | residual RMS (% of mix) | residual dB | sum-to-mix SDR | residual peak |
| --- | ---: | ---: | ---: | ---: |
| Base BS-RoFormer | 4.947953% | -26.11 dB | 26.11 dB | 0.24482233 |
| Base + Residual Allocator | 0.000005% | -146.01 dB | 146.01 dB | 0.00000024 |

## Application Example: Stem Upmix

`examples/stem_upmix/upmix.py` is an optional example script that analyzes
separated stems and renders a simple music-oriented upmix. It expects a stem
directory containing files such as `vocals.wav`, `drums.wav`, `bass.wav`,
`guitar.wav`, `piano.wav`, and `other.wav`.

Install `ffmpeg` and `ffprobe` first; they must be available on `PATH`.

```bash
python examples/stem_upmix/upmix.py \
  --stem-dir outputs/song \
  --input /path/to/song.wav
```

By default, the script writes outputs under `<stem-dir>/upmix`:

- a 7.1.4 WAV bed
- a 5.1 FLAC fold-down
- a stereo FLAC fold-down
- an Apple TV compatible E-AC-3 MP4

Useful options:

- `--output-dir`: write upmix files to a specific directory.
- `--skip-apple-tv`: skip the Apple TV MP4 render.
- `--skip-stereo`: skip the stereo FLAC fold-down.
- `--skip-bed`: remove the intermediate 7.1.4 WAV after derived outputs are rendered.
- `--reference-match original`: match the stereo fold-down balance against the original input.

## License

Residual Allocator is licensed under the Apache License, Version 2.0. See
`LICENSE` and `NOTICE`.

This repository also includes BS-RoFormer code adapted from
[lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer), which is
licensed under the MIT License. See `THIRD_PARTY_NOTICES.md` for the upstream
copyright and license notice.
