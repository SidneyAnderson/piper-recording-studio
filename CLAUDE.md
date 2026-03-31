# Claude Code Context — Piper Recording Studio

## Project Overview

This is a fork of [rhasspy/piper-recording-studio](https://github.com/rhasspy/piper-recording-studio) extended with:
- **ElevenLabs TTS integration** — generate training audio from prompts via ElevenLabs API
- **Prompt generator** — create 12k+ phonetically diverse training sentences
- **Full training pipeline** — web UI for the entire workflow: generate → export → setup → preprocess → train → test → export ONNX
- **Voice profiles** — save/load ElevenLabs voice configurations
- **Checkpoint browser** — download pre-trained checkpoints by locale and quality tier

Fork: https://github.com/SidneyAnderson/piper-recording-studio
Branch: `elevenlabs-integration`

## Architecture

- **Web server**: Quart (async Python) + Hypercorn, templates in `piper_recording_studio/templates/`
- **ElevenLabs client**: `elevenlabs_generate/` — async httpx client, CLI batch generator
- **Prompt generator**: `generate_prompts/` — 18 categories, phonetically diverse
- **Training**: Uses Piper's training code in `piper/src/python/` (cloned by setup script)
- **Checkpoint catalog**: `train/checkpoints.py` — 30+ languages
- **Piper patches**: `train/patch_piper.py` — PyTorch 2.x compatibility

## Key Technical Details

### Training Pipeline (CRITICAL — hard-won knowledge)

1. **PyTorch 2.x + pytorch-lightning 1.7**: Compatible! Automatic optimization works. The `optimizer_idx` IS passed correctly by PL 1.7.

2. **Checkpoint saving**: PL's default `ModelCheckpoint` has `save_top_k=1` which DELETES old checkpoints. We use `enable_checkpointing=False` + custom `SimpleCheckpoint` callback (defined in `piper/src/python/piper_train/__main__.py`) that keeps all checkpoints.

3. **DO NOT use `trainer.callbacks = [...]`** — this REPLACES all PL internal callbacks and breaks the training loop. Always use `trainer.callbacks.append()`.

4. **Patches applied by `train/patch_piper.py`**:
   - `torch.serialization.add_safe_globals` for PosixPath (PyTorch 2.6+)
   - `lr_scheduler_step` override (PyTorch 2.x LR scheduler API)
   - `SimpleCheckpoint` callback + `enable_checkpointing=False`
   - Legacy ONNX exporter (`dynamo=False`)

5. **Batch sizes**: High quality model OOMs at batch 64 on 32GB VRAM. Use batch 32 for high, 64 for medium.

6. **First epoch**: Takes 5-10 minutes (CUDA kernel compilation). NOT stuck — this is normal.

7. **Training time**: ~26s per epoch at batch 32 high quality. 500 epochs ≈ 12-15 hours.

8. **Training process**: Runs detached (`start_new_session=True`), survives server restarts. PID saved in `training.pid`. Stop via web UI or `pkill -f piper_train`.

9. **Training metadata**: `training.pid` stores five lines: PID, start_time, requested_epochs, ckpt_base_epoch, max_epochs. The helper `_read_training_pid_file()` returns a **dict** (not a tuple) with keys: `pid`, `start_time`, `requested_epochs`, `ckpt_base_epoch`, `max_epochs`.

10. **Stats endpoint** (`/api/training/stats`): Returns `loss_gen`, `loss_disc` arrays with `step`, `value`, and `epoch` fields, plus `checkpoint_epochs` (list of epoch numbers that have saved checkpoints).

11. **Charts**: Loss charts are aligned to checkpoint epochs and split into separate Generator/Discriminator tabs in the frontend. Epoch labels on the x-axis are user-facing (1 to N), calculated by subtracting `ckpt_base_epoch`.

12. **`args.enable_checkpointing = False`**: Must be set directly on the `Namespace` object because `Trainer.from_argparse_args` gives Namespace values priority over kwargs.

### Setup Script Dependencies (fragile — order matters)

```
pip<24.1           # Required — PL 1.7 metadata is invalid in pip>=24.1
numpy<2            # Required — PyTorch compiled against NumPy 1.x
torchmetrics==0.11.4  # Required — newer versions break PL 1.7
six                # Required — tensorboard dependency
onnxscript         # Required — PyTorch 2.x ONNX export
torch+cu128        # Required — RTX 5090 needs CUDA 12.8 (sm_120)
```

### Web UI Pages

- `/` — Home (record, generate, train buttons)
- `/generate` — ElevenLabs TTS generation with voice profiles
- `/train` — Full training pipeline (7 steps, all browser-based)
- `/record` — Original Piper recording interface

### Files NOT in git (gitignored)

- `piper/` — Cloned Piper repo (patched by setup script)
- `output/` — Generated audio files
- `dataset_*/` — Exported LJSpeech datasets
- `training/` — Preprocessed training data + lightning_logs
- `checkpoints/` — Downloaded pre-trained checkpoints
- `models/` — Exported ONNX models
- `voice_profiles/` — Saved voice configs
- `.env` — ElevenLabs API key

### Current State (as of 2026-03-31)

- 1,150 audio samples generated via ElevenLabs (en-GB, British voice)
- 11,000 additional prompts generated (not yet sent through ElevenLabs)
- Training works — verified with automatic optimization
- User's hardware: RTX 5090 (32GB), WSL2 Ubuntu 22.04
- ElevenLabs voice: st7NwhTPEzqo2riw7qWC (Voice Library British voice)
- Checkpoint: en_GB-cori-high (~952MB)
