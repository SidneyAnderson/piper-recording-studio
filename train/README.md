# Training a Piper Voice from Your Dataset

This guide walks you through turning your exported dataset into a usable Piper TTS voice.

## Overview

| Step | What it does | Time |
|------|-------------|------|
| 1. Export | Convert recordings to LJSpeech format | Minutes |
| 2. Setup | Install Piper training environment | 10-15 min |
| 3. Download checkpoint | Get a pre-trained model to fine-tune from | Minutes |
| 4. Preprocess | Convert audio to training tensors | Minutes |
| 5. Train | Fine-tune the voice model on GPU | Hours to days |
| 6. Export | Convert checkpoint to .onnx for inference | Seconds |
| 7. Test | Generate speech with your new voice | Seconds |


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

Visit http://localhost:8000/train and use the **checkpoint browser** in Step 3. Select your language, voice, and quality tier — then click Download. The UI streams download progress.

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
- The voice name in the checkpoint doesn't matter much — fine-tuning will adapt it to your dataset


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

Visit http://localhost:8000/train, configure batch size and epochs, and click **Start Training**. Training log streams live in the browser. You can stop and resume at any time.

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
  --checkpoint-epochs 10 \
  --precision 32
```

### Adjusting batch size for your GPU

| GPU VRAM | Recommended batch size |
|----------|----------------------|
| 8 GB | 12 |
| 10-12 GB | 24 |
| 16 GB | 32 |
| 24+ GB | 32-64 |

Larger batch sizes train faster but don't affect final model quality. If you get out-of-memory errors, reduce `--batch-size`.

### Monitoring training

In a separate terminal:

```sh
tensorboard --logdir /path/to/training/lightning_logs
```

Visit http://localhost:6006 to watch loss curves. Key metrics:
- `loss_disc_all` — should decrease and then plateau
- Stop training when the loss plateaus (typically 500-1000 epochs for fine-tuning)

### Disk space

Training checkpoints are ~400 MB each (high quality: ~1 GB). With `--checkpoint-epochs 10` and 1000 epochs, that's roughly 40-100 GB. Manage disk space by:
- Increasing `--checkpoint-epochs` to save less frequently
- Periodically deleting old checkpoints you've already tested
- Keeping at least 10-50 GB free


## Step 6: Export to ONNX

### Option A: Web UI

Visit http://localhost:8000/train, select a checkpoint from the dropdown in Step 6, and click **Export Selected Checkpoint**.

### Option B: CLI

```sh
cd piper/src/python
source .venv/bin/activate

python3 -m piper_train.export_onnx \
  /path/to/training/lightning_logs/version_0/checkpoints/CHECKPOINT_FILE.ckpt \
  /path/to/my_voice.onnx

# IMPORTANT: the config file MUST accompany the model
cp /path/to/training/config.json /path/to/my_voice.onnx.json
```

Both files (`.onnx` and `.onnx.json`) are required — the model will not work without the config.


## Step 7: Test Your Voice

### Option A: Web UI

Visit http://localhost:8000/train, type text in Step 7, and click **Test Voice** to hear it in the browser. Requires `piper-tts` to be installed.

### Option B: CLI

```sh
pip install piper-tts

echo "Hello, this is my custom voice!" | piper -m /path/to/my_voice.onnx --output_file test.wav

# Play it
aplay test.wav
# or
ffplay test.wav
```


## Troubleshooting

| Problem | Solution |
|---------|----------|
| `pytorch-lightning` install fails | Use `pip<24.1` in the training venv (the setup script handles this) |
| Out of memory during training | Reduce `--batch-size` |
| Training loss not decreasing | Check that your audio quality is consistent and clean |
| Exported model sounds robotic | You may have over-trained — try an earlier checkpoint |
| `piper` command not found | Run `pip install piper-tts` in your main venv |
| Checkpoint download fails | Checkpoints are public, no auth needed. Check your internet connection. |
| `espeak-ng` not found | Run `sudo apt-get install espeak-ng` |


## Tips

- **Quality over quantity:** Clean, consistent audio (like ElevenLabs output) is ideal for training. 1,000+ samples with fine-tuning should produce good results.
- **Don't over-train:** More epochs is not always better. Over-training causes distortion. Monitor TensorBoard and stop when loss plateaus.
- **Test during training:** You can export and test any checkpoint while training continues. Try one every 100 epochs to find the sweet spot.
- **pip version matters:** Piper's `pytorch-lightning~=1.7.0` dependency has invalid metadata that pip>=24.1 rejects. Always use `pip<24.1` in the training venv.
- **Training from scratch:** Not recommended with under 5,000 samples. If you must, expect to need 2,000+ epochs and the quality may not match fine-tuning.
- **Resumable:** Training can be stopped and resumed. The web UI and CLI both support `--resume_from_checkpoint` to continue from the latest saved checkpoint.
