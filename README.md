# Residual Allocator

Residual Allocator is a lightweight post-refinement module for BS-RoFormer stem
separation. It routes the residual between the input mix and base separator
outputs back into the most plausible stems.

This repository contains the inference code, model definition, base BS-RoFormer
architecture code needed for loading checkpoints, and a ConvResidualAllocator
inference checkpoint.

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

## Example: Upmix Separated Stems

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
