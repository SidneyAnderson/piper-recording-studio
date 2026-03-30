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

- **NVIDIA GPU** with at least 8 GB VRAM (e.g., GTX 1080, RTX 3060+)
- NVIDIA drivers + CUDA installed
- Docker (recommended) or Python 3.10 with pip
- `espeak-ng` installed: `sudo apt-get install espeak-ng`
- Your dataset exported to LJSpeech format (see Step 1)


## Step 1: Export Your Dataset

If you haven't already exported:

```sh
cd /path/to/piper-recording-studio
python3 -m export_dataset --audio-glob '*.wav' output/en-GB/ dataset_en-GB/
```

Verify: `wc -l dataset_en-GB/metadata.csv` should show your prompt count.


## Step 2: Set Up Piper Training

```sh
git clone https://github.com/rhasspy/piper.git
cd piper/src/python

python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip wheel setuptools
pip3 install -e .

# Build required C extension
bash build_monotonic_align.sh

# Install espeak-ng if not already installed
sudo apt-get install espeak-ng
```


## Step 3: Download a Pre-Trained Checkpoint

Fine-tuning from an existing checkpoint is **strongly recommended** for datasets under 5,000 samples. It produces much better results than training from scratch.

Pre-trained checkpoints are available at:
https://huggingface.co/datasets/rhasspy/piper-checkpoints

For English (medium quality, recommended starting point):

```sh
# Create a directory for checkpoints
mkdir -p /path/to/checkpoints

# Download the English US "lessac" medium checkpoint (~400 MB)
wget -O /path/to/checkpoints/en_US-lessac-medium.ckpt \
  "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt"
```

### Quality tiers

| Tier | Sample rate | Model size | Notes |
|------|------------|------------|-------|
| Low | 16000 Hz | Smallest | Fastest inference, lower quality |
| Medium | 22050 Hz | Mid | Good balance of speed and quality |
| High | 22050 Hz | Largest | Best quality, slower inference |

**Important:** Your audio sample rate must match the checkpoint tier. ElevenLabs-generated audio at 24000 Hz will be resampled during preprocessing.


## Step 4: Preprocess

```sh
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

```sh
python3 -m piper_train \
  --dataset-dir /path/to/training/ \
  --accelerator gpu \
  --devices 1 \
  --batch-size 32 \
  --validation-split 0.0 \
  --num-test-examples 0 \
  --max_epochs 1000 \
  --resume_from_checkpoint /path/to/checkpoints/en_US-lessac-medium.ckpt \
  --checkpoint-epochs 1 \
  --precision 32
```

### Adjusting for your GPU

| GPU VRAM | Recommended batch size |
|----------|----------------------|
| 8 GB | 12 |
| 10-12 GB | 24 |
| 16+ GB | 32 |
| 24 GB | 32-48 |

If you get out-of-memory errors, reduce `--batch-size`.

### Monitoring training

In a separate terminal:

```sh
tensorboard --logdir /path/to/training/lightning_logs
```

Visit http://localhost:6006 to watch loss curves. Key metrics:
- `loss_disc_all` — should decrease and then plateau
- Stop training when the loss plateaus (typically 500-1000 epochs for fine-tuning)

### Checkpoints

Checkpoints are saved to `training/lightning_logs/version_0/checkpoints/` every epoch (~400 MB each). You can test any checkpoint without stopping training.


## Step 6: Export to ONNX

Once training is done (or you want to test a checkpoint):

```sh
python3 -m piper_train.export_onnx \
  /path/to/training/lightning_logs/version_0/checkpoints/CHECKPOINT_FILE.ckpt \
  /path/to/my_voice.onnx

# IMPORTANT: copy the config alongside the model
cp /path/to/training/config.json /path/to/my_voice.onnx.json
```


## Step 7: Test Your Voice

Install Piper for inference:

```sh
pip install piper-tts
```

Generate speech:

```sh
echo "Hello, this is my custom voice!" | piper -m /path/to/my_voice.onnx --output_file test.wav
```

Play the output:

```sh
aplay test.wav
# or
ffplay test.wav
```


## Tips

- **Quality over quantity:** Your ElevenLabs-generated audio is clean and consistent, which is ideal for training. 1,150 samples with fine-tuning should produce good results.
- **Don't over-train:** More epochs is not always better. Over-training causes distortion. Monitor TensorBoard and stop when loss plateaus.
- **Save disk space:** Checkpoints are ~400 MB each. With `--checkpoint-epochs 1` and 1000 epochs, that's ~400 GB. Consider `--checkpoint-epochs 10` or periodically delete old checkpoints.
- **Test during training:** You can export and test any checkpoint while training continues. Try one every 100 epochs to find the sweet spot.
- **Training from scratch:** Not recommended with under 5,000 samples. If you must, expect to need 2,000+ epochs and the quality may not match fine-tuning.
