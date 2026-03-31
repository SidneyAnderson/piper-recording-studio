# Training a Piper Voice from Your Dataset

This guide walks you through turning your exported dataset into a usable Piper TTS voice.

## Overview

| Step | What it does | Time |
|------|-------------|------|
| 1. Export dataset | Convert recordings to LJSpeech format | Minutes |
| 2. Setup | Install Piper training environment | 10-15 min |
| 3. Download checkpoint | Get a pre-trained model to fine-tune from | Minutes |
| 4. Preprocess | Convert audio to training tensors | Minutes |
| 5. Train | Fine-tune the voice model on GPU | 10-50 hours (see estimates below) |
| 6. Test | Listen to checkpoints and find the best epoch | Seconds |
| 7. Export model | Save the best checkpoint as a portable .onnx file | Seconds |


## Prerequisites

- **NVIDIA GPU** with at least 8 GB VRAM (more VRAM allows larger batch sizes and faster training)
- NVIDIA drivers + CUDA installed
- Python 3.10 with pip
- System packages: `sudo apt-get install ffmpeg espeak-ng`
- Your dataset exported to LJSpeech format (see Step 1)
- No HuggingFace account needed — all checkpoints are publicly available


## Step 1: Export Your Dataset

If you used the ElevenLabs generator, your audio files are WAV. Export with the `--audio-glob` flag:

```sh
cd /path/to/piper-recording-studio

# Install export dependencies
pip install -r requirements_export.txt

# Export (note: --audio-glob '*.wav' is required for ElevenLabs-generated audio)
python3 -m export_dataset --audio-glob '*.wav' output/en-GB/ dataset_en-GB/
```

Verify: `wc -l dataset_en-GB/metadata.csv` should show your prompt count.


## Step 2: Set Up Piper Training

**Option A: Automated setup (recommended)**

```sh
bash setup_training.sh
```

This handles steps 2-4 automatically: clones Piper, installs dependencies, downloads the checkpoint, and preprocesses your dataset.

**Option B: Manual setup**

```sh
git clone https://github.com/rhasspy/piper.git
cd piper/src/python

python3 -m venv .venv
source .venv/bin/activate

# IMPORTANT: Piper requires pytorch-lightning~=1.7.0 which needs pip<24.1
pip3 install "pip<24.1" wheel setuptools
pip3 install -e .

# Build required C extension
bash build_monotonic_align.sh
```


## Step 3: Download a Pre-Trained Checkpoint

Fine-tuning from an existing checkpoint is **strongly recommended** for datasets under 5,000 samples. It produces much better results than training from scratch.

All checkpoints are publicly available — **no HuggingFace account or token is required**.

### Option A: Web UI (recommended)

Visit http://localhost:8000/train and use the **checkpoint browser** in Step 3. Select your locale and quality tier — the best checkpoint is auto-selected. Click Download. The UI streams download progress.

### Option B: CLI

Pre-trained checkpoints are available at:
https://huggingface.co/datasets/rhasspy/piper-checkpoints

```sh
mkdir -p checkpoints

# Example: Download English GB "cori" high quality checkpoint (~1 GB)
wget -O checkpoints/en_GB-cori-high.ckpt \
  "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_GB/cori/high/cori-high-500.ckpt"
```

### Quality tiers

| Tier | Sample rate | Model size | Notes |
|------|------------|------------|-------|
| Low | 16,000 Hz | ~200 MB | Fastest inference, basic quality |
| Medium | 22,050 Hz | ~400 MB | Good balance of speed and quality |
| **High** | **22,050 Hz** | **~1 GB** | **Best quality (recommended when GPU allows)** |

**Important:** Your audio sample rate must match the checkpoint tier. ElevenLabs-generated audio at 24000 Hz will be resampled automatically during preprocessing.

### Choosing the right checkpoint

- Pick a checkpoint in the **same language** as your training data for best results
- **High quality** produces the best output and is recommended if your GPU has 16+ GB VRAM
- The checkpoint's original voice doesn't matter — fine-tuning replaces it with your dataset's voice


## Step 4: Preprocess

```sh
cd piper/src/python
source .venv/bin/activate

python3 -m piper_train.preprocess \
  --language en \
  --input-dir /path/to/dataset_en-GB/ \
  --output-dir /path/to/training/ \
  --dataset-format ljspeech \
  --single-speaker \
  --sample-rate 22050
```

This creates `config.json`, `dataset.jsonl`, and audio tensor files.


## Step 5: Train (Fine-Tune)

### Option A: Web UI (recommended)

Visit http://localhost:8000/train, configure batch size and epochs, and click **Start Training**. The UI shows a live progress bar with elapsed time and ETA, plus a Statistics tab with loss curves. Training survives browser refreshes and server restarts.

**Recommended settings for a first run:** batch size 64, 500 epochs, save every 25.

### Option B: CLI

```sh
cd piper/src/python
source .venv/bin/activate

python3 -m piper_train \
  --dataset-dir /path/to/training/ \
  --accelerator gpu \
  --devices 1 \
  --batch-size 32 \
  --validation-split 0.0 \
  --num-test-examples 0 \
  --max_epochs 1000 \
  --resume_from_checkpoint /path/to/checkpoints/YOUR_CHECKPOINT.ckpt \
  --checkpoint-epochs 25 \
  --precision 32 \
  --quality high
```

### Training time estimates

| Epochs | Batch Size | Quality | Approx. Time |
|--------|-----------|---------|-------------|
| 500 | 32 | High | ~12-15 hours |
| 1000 | 32 | High | ~25-30 hours |
| 500 | 64 | Medium | ~6-8 hours |
| 500 | 32 | Medium | ~10-12 hours |

Piper uses full-model retraining (not LoRA/adapter fine-tuning), which is why training takes longer than typical image or video fine-tuning. The dual generator+discriminator architecture (GAN-style) effectively trains two models per batch.

### First epoch and CUDA compilation

The first training epoch takes **5-10 minutes** due to CUDA kernel compilation (PyTorch 2.x). This is normal — subsequent epochs run much faster (~2-7 seconds depending on batch size and quality tier). Don't assume training is stuck during the first epoch.

### Batch size by GPU VRAM

| VRAM | Batch Size | Notes |
|------|-----------|-------|
| 8 GB | 12 | Slowest, but works |
| 10-12 GB | 24 | |
| 16 GB | 32 | |
| 24+ GB | 32 (high) / 64 (medium) | High quality model uses more VRAM per batch |

Larger batch sizes train faster but don't affect final model quality. If you get out-of-memory errors, reduce `--batch-size`.

### Monitoring training

The web UI at `/train` includes a **Statistics tab** that shows live loss curves (generator and discriminator) during training. Both losses should decrease and plateau — stop training when they flatten out.

Key metrics:
- `loss_gen_all` (generator loss) — how well the model generates speech
- `loss_disc_all` (discriminator loss) — how well the model distinguishes real from generated speech
- Stop training when both losses plateau (typically 300-500 epochs for fine-tuning)
- Early oscillation is normal in the first 100-200 epochs

### Disk space

Training checkpoints are ~400 MB each (high quality: ~1 GB). With `--checkpoint-epochs 10` and 1000 epochs, that's roughly 40-100 GB. Manage disk space by:
- Increasing `--checkpoint-epochs` to save less frequently
- Periodically deleting old checkpoints you've already tested
- Keeping at least 10-50 GB free


## Step 6: Test Your Voice

Select a training checkpoint and listen to it. Try different epochs to find the best sounding one before exporting.

### Web UI

Visit http://localhost:8000/train. In Step 6, select a checkpoint from the dropdown, click **Load & Test This Epoch**, type sample text, and click **Play Voice**.

### CLI

```sh
# Export a checkpoint temporarily for testing
cd piper/src/python && source .venv/bin/activate
python3 -m piper_train.export_onnx \
  /path/to/training/lightning_logs/version_0/checkpoints/epoch=509-step=775720.ckpt \
  /tmp/test_voice.onnx
cp /path/to/training/config.json /tmp/test_voice.onnx.json

# Listen
echo "Hello, this is my custom voice!" | piper -m /tmp/test_voice.onnx --output_file test.wav
aplay test.wav
```


## Step 7: Export Final Model

Once you've found the best sounding epoch in Step 6, export it as your final model.

### Web UI

In Step 7, select the checkpoint and click **Export Selected Checkpoint**. The model is saved as `models/<profile_name>.<epoch>.onnx` (e.g. `models/British_Narrator.510.onnx`).

### CLI

```sh
mkdir -p models

python3 -m piper_train.export_onnx \
  /path/to/training/lightning_logs/version_0/checkpoints/epoch=509-step=775720.ckpt \
  models/My_Voice.510.onnx

# IMPORTANT: the config file MUST accompany the model
cp /path/to/training/config.json models/My_Voice.510.onnx.json
```

Both files (`.onnx` and `.onnx.json`) are required — the model will not work without the config.

### Using your model

```sh
echo "Hello world!" | piper -m models/British_Narrator.510.onnx --output_file test.wav
```


## PyTorch 2.x Compatibility

Piper's training code was written for PyTorch 1.x and pytorch-lightning 1.7. Modern GPUs (RTX 40xx/50xx) require PyTorch 2.6+ with CUDA 12.8, which introduces several breaking changes. The setup script automatically applies patches via `train/patch_piper.py`:

1. **Manual optimization** — PyTorch 2.x removed the `optimizer_idx` parameter from `training_step`. The patch switches to manual optimization where the generator and discriminator are trained explicitly.
2. **Custom checkpoint callback** — `ModelCheckpoint` from pytorch-lightning 1.7 is broken with PyTorch 2.x. A custom `SimpleCheckpoint` callback handles checkpoint saving.
3. **Safe globals for checkpoint loading** — PyTorch 2.6+ requires explicit allowlisting of `pathlib.PosixPath` for loading older checkpoints.
4. **Legacy ONNX exporter** — PyTorch 2.6+ defaults to a dynamo-based ONNX exporter that's incompatible with VITS. The patch forces the legacy exporter.

These patches are applied automatically by `bash setup_training.sh` and are transparent to the user.


## Troubleshooting

| Problem | Solution |
|---------|----------|
| `pytorch-lightning` install fails | Use `pip<24.1` in the training venv (the setup script handles this) |
| Out of memory during training | Reduce batch size. High quality at batch 64 needs ~24 GB VRAM. |
| First epoch takes 5-10 minutes | Normal — CUDA kernel compilation on PyTorch 2.x. Subsequent epochs are fast. |
| Training loss not decreasing | Check that your audio quality is consistent and clean |
| Exported model sounds robotic | You may have over-trained — try an earlier checkpoint |
| `piper` command not found | Included in `requirements.txt` — run `pip install -r requirements.txt` |
| Checkpoint download fails | Checkpoints are public, no auth needed. Check your internet connection. |
| `espeak-ng` not found | Run `sudo apt-get install espeak-ng` |
| CUDA error: no kernel image | Your PyTorch doesn't support your GPU. Run `setup_training.sh` to install PyTorch with CUDA 12.8. |
| `optimizer_idx` or `ExponentialLR` error | Run `setup_training.sh` to apply PyTorch 2.x patches. |
| Zero checkpoints saved | Run `setup_training.sh` to apply the custom checkpoint callback patch. |


## Tips

- **Quality over quantity:** Clean, consistent audio (like ElevenLabs output) is ideal for training. 1,000+ samples with fine-tuning should produce good results.
- **Don't over-train:** More epochs is not always better. Over-training causes distortion. Watch the Statistics tab on the training page — stop when losses plateau.
- **Test during training:** You can export and test any checkpoint while training continues. Try one at epoch 250 and 500 to find the sweet spot.
- **Batch size 64:** If your GPU has 24+ GB VRAM, use batch size 64 to cut training time roughly in half.
- **Training from scratch:** Not recommended with under 5,000 samples. If you must, expect to need 2,000+ epochs and the quality may not match fine-tuning.
- **Resumable:** Training can be stopped and resumed. The web UI automatically resumes from the latest saved checkpoint.
- **Training is full-model:** Unlike image LoRA, Piper retrains the full 83M parameter model (generator + discriminator). This is why training takes hours, not minutes.
