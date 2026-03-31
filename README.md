# Piper Recording Studio

Local tool for recording yourself to train a [Piper text to speech](https://github.com/rhasspy/piper) voice.

![Screen shot](etc/screenshot.jpg)

[![Sponsored by Nabu Casa](etc/nabu_casa_sponsored.png)](https://nabucasa.com)


## Tutorial

See a [video tutorial](https://www.youtube.com/watch?v=Z1pptxLT_3I) by [Thorsten Müller](https://www.thorsten-voice.de/)


## Docker

``` sh
docker run -it -p 8000:8000 -v '/path/to/output:/app/output' rhasspy/piper-recording-studio
```

Visit http://localhost:8000 to select a language and start recording.

Add `--help` to see more options.

Note: Docker is suitable for recording and ElevenLabs generation. Training requires a native install with GPU access.


### Building

``` sh
docker build . -t rhasspy/piper-recording-studio
```


## Installing without Docker

### System dependencies

``` sh
sudo apt-get install ffmpeg espeak-ng
```

### Python environment

``` sh
git clone https://github.com/rhasspy/piper-recording-studio.git
cd piper-recording-studio/

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Additional dependencies by feature

| Feature | Requirements file | System packages |
|---------|------------------|-----------------|
| Web UI + ElevenLabs | `requirements.txt` | — |
| Dataset export | `requirements_export.txt` | `ffmpeg` |
| Training | Installed by `setup_training.sh` | `ffmpeg`, `espeak-ng`, NVIDIA CUDA |
| Voice testing | Included in `requirements.txt` | — |


## Running without Docker

``` sh
python3 -m piper_recording_studio
```

Visit http://localhost:8000 to select a language and start recording.

Prompts are in the `prompts/` directory with the following format:

* Language directories are named `<language name>_<language code>`
* Each `.txt` in a language directory contains lines with:
    * `<id>\t<text>` or
    * `text` (id is automatically assigned based on line number)

Output audio is written to `output/`

See `--debug` for more options.


## Exporting

Export recordings for a language to a Piper-compatible dataset (LJSpeech format):

``` sh
# Install export dependencies
python3 -m pip install -r requirements_export.txt

# Export (use --audio-glob '*.wav' for ElevenLabs-generated audio)
python3 -m export_dataset --audio-glob '*.wav' output/<language>/ dataset_<language>/
```

If you used Docker to record your dataset, you may need to adjust the permissions of the output directory:

``` sh
sudo chown -R "$(id -u):$(id -u)" output/
```

See `--help` for more options. You may need to adjust the silence detection parameters to correctly remove button clicks and keypresses.


## Generating Training Prompts

For high-quality from-scratch voice training, you need 10,000+ diverse sentences. The built-in prompt generator creates phonetically rich, natural-sounding text optimized for TTS training.

``` sh
# Generate 11,000 prompts for British English
python3 -m generate_prompts --count 11000 --language en-GB
```

The generator follows ElevenLabs best practices:
- Sentences between 10-30 words for optimal TTS quality
- Numbers written as words, abbreviations expanded
- All English phonemes covered through natural vocabulary
- 18 topic categories for diversity (daily life, travel, science, British culture, etc.)
- Mix of prosodic patterns (statements, questions, commands, exclamations)
- Both British and General English phrasing

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--count` | `11000` | Number of prompts to generate |
| `--language` | `en-GB` | Language code for output directory |
| `--seed` | `42` | Random seed for reproducibility |
| `--categories` | all | Specific categories to generate |
| `--dry-run` | — | Print to stdout instead of writing file |

Generated prompts are saved to `prompts/<Language>_<code>/` in the format expected by the recording studio and ElevenLabs generator.

### Recommended ElevenLabs settings for training data

For maximum consistency when generating a large dataset:

| Setting | Recommended | Why |
|---------|------------|-----|
| Stability | 0.75-0.85 | Higher values produce more consistent, predictable output |
| Similarity Boost | 0.80-1.00 | Keeps output closer to the voice profile |
| Sample Rate | 24000 Hz | Good balance of quality and compatibility |


## ElevenLabs TTS Generation

Generate audio from prompts using the [ElevenLabs](https://elevenlabs.io/) API instead of recording manually. Useful for creating synthetic training datasets.

### Requirements

An ElevenLabs account with an API key is required. Create one at https://elevenlabs.io/ under **Developers > API Keys**. The key needs at minimum:

* **Text to Speech** — Access
* **Models** — Access
* **Voices** — Read (for automatic voice/model detection)
* **User** — Read (for API key validation)

### Web UI

Start the recording studio as normal, then click **ElevenLabs TTS** on the home page (or visit http://localhost:8000/generate).

Features:

* **API Key Test** — validates your key and shows tier/character usage
* **Voice ID lookup** — enter a voice ID and the app automatically detects the voice name and selects the correct model
* **Model dropdown** — populated from ElevenLabs with compatible models marked as "(recommended)"
* **Voice preview** — generate and play a single sample to tune stability, similarity boost, and sample rate before committing to a full run
* **Real-time progress** — SSE streaming with progress bar and scrollable log
* **Auto-retry** — transient network errors retry up to 10 times with increasing backoff
* **Auto-reconnect** — if the browser connection drops, it reconnects and resumes automatically (up to 5 times)
* **Resumable** — stop and restart at any time; completed prompts are skipped
* **Persistent config** — API key, voice ID, and model ID are saved to `.env` and auto-loaded on next visit

Note: 24000 Hz is the recommended sample rate for Piper training. Higher rates (44100 Hz) are not used by Piper and have been removed from the UI.

### CLI

``` sh
python3 -m elevenlabs_generate \
  --api-key YOUR_API_KEY \
  --voice-id YOUR_VOICE_ID \
  --model-id eleven_multilingual_v2 \
  --language en-US
```

The API key can also be set via the `ELEVENLABS_API_KEY` environment variable.

Test your API key without generating:

``` sh
python3 -m elevenlabs_generate --api-key YOUR_API_KEY --test
```

Additional options:

| Flag | Default | Description |
|------|---------|-------------|
| `--prompts` | `prompts/` | Path to prompts directory |
| `--output` | `output/` | Path to output directory |
| `--sample-rate` | `24000` | Audio sample rate (24000 recommended for Piper) |
| `--stability` | `0.5` | Voice stability (0.0–1.0) |
| `--similarity-boost` | `0.75` | Voice similarity boost (0.0–1.0) |
| `--rate-limit-delay` | `0.5` | Seconds between API calls |
| `--test` | — | Validate API key and exit |
| `--debug` | — | Enable debug logging |

Generation is **resumable** — it skips prompts that already have output files. Audio is saved as WAV in the same `output/<language>/` structure used by the recorder, so the existing export pipeline works without changes.

### Configuration (.env)

On first successful API key test, the app saves your configuration to a `.env` file in the project root:

```
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_VOICE_ID=your_voice_id
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
```

This file is gitignored and auto-loaded when you open the generate page. The CLI also accepts these as environment variables.


## Training a Piper Voice

After generating audio, use the built-in training guide to create a custom Piper TTS voice.

### Prerequisites

- **NVIDIA GPU** with at least 8 GB VRAM (more VRAM allows larger batch sizes and faster training)
- NVIDIA drivers + CUDA installed
- System packages: `sudo apt-get install ffmpeg espeak-ng`
- No HuggingFace account required — all checkpoints are publicly available

### Quick start

``` sh
# 1. Export dataset (--audio-glob '*.wav' required for ElevenLabs audio)
pip install -r requirements_export.txt
python3 -m export_dataset --audio-glob '*.wav' output/en-GB/ dataset_en-GB/

# 2. Run the automated setup (clones Piper, downloads checkpoint, preprocesses)
bash setup_training.sh

# 3. Start training via the web UI or command line
python3 -m piper_recording_studio
# Visit http://localhost:8000/train and click 'Start Training'
```

The setup script handles:
- Cloning the Piper repo and installing dependencies (with pinned pip<24.1 for compatibility)
- Installing PyTorch with CUDA 12.8 for modern GPUs (RTX 40xx/50xx)
- Applying PyTorch 2.x compatibility patches (manual optimization, checkpoint saving, ONNX export)
- Downloading a pre-trained checkpoint from HuggingFace (no auth required)
- Preprocessing your dataset into training-ready tensors

### Web UI guide

Visit http://localhost:8000/train (or click **Training Guide** on the home page) for the full pipeline:

* **Dataset status** — shows generated files, export status, and preprocessing state
* **Checkpoint browser** — select locale and quality tier from 30+ languages; best checkpoint auto-selected; download with progress streaming
* **Training launcher** — start/stop training with configurable batch size, epochs, and checkpoint frequency; live progress bar and log streaming
* **Voice testing** — select any training checkpoint, load it, type text, and listen in the browser to find the best epoch
* **Model export** — export the best checkpoint as `profile_name.epoch.onnx` to the `models/` directory

### Selecting a checkpoint

The web UI includes a **checkpoint browser** — select your locale and quality tier (low/medium/high). The best checkpoint is auto-selected. The **high quality tier** (~1 GB) is recommended when your GPU has 16+ GB VRAM.

Fine-tuning from a pre-trained checkpoint produces significantly better results than training from scratch with fewer than 5,000 samples. The checkpoint provides existing knowledge of speech patterns — training only adapts the voice characteristics.

All checkpoints sourced from: https://huggingface.co/datasets/rhasspy/piper-checkpoints (public, no auth required)

### Training time estimates

| Epochs | Batch Size | Quality | Approx. Time |
|--------|-----------|---------|-------------|
| 500 | 32 | High | ~12-15 hours |
| 1000 | 32 | High | ~25-30 hours |
| 500 | 64 | Medium | ~6-8 hours |
| 500 | 32 | Medium | ~10-12 hours |

The first epoch takes 5-10 minutes due to CUDA kernel compilation (PyTorch 2.x). After that, each epoch takes 2-7 seconds depending on batch size and model quality. This is full-model retraining (not LoRA/adapter), which is why it takes longer than image fine-tuning.

**Recommended first run:** 500 epochs, batch size 32 (high quality) or 64 (medium), save every 25. Test the voice at epoch 250 — if it sounds good, you can stop early.

### Batch size by GPU VRAM

| VRAM | Batch Size | Notes |
|------|-----------|-------|
| 8 GB | 12 | Slowest, but works |
| 10-12 GB | 24 | |
| 16 GB | 32 | |
| 24+ GB | 32 (high) / 64 (medium) | High quality model uses more VRAM per batch |

Larger batch sizes train faster but don't affect final model quality. If you get out-of-memory errors, reduce the batch size.

### Disk space

Training checkpoints are ~400 MB each (high quality: ~1 GB). With save every 25 epochs over 500 epochs, that's ~20 checkpoints = 20 GB for high quality. Plan accordingly.

### PyTorch 2.x compatibility

Piper's training code was designed for PyTorch 1.x. The setup script automatically applies patches for PyTorch 2.x compatibility:
- Manual optimization for multi-optimizer training (generator + discriminator)
- Custom checkpoint callback (replaces broken ModelCheckpoint)
- Safe globals for checkpoint loading (PyTorch 2.6+)
- Legacy ONNX exporter (PyTorch 2.6+ dynamo fix)

These patches are applied by `train/patch_piper.py` during setup and are transparent to the user.

### Testing your trained voice

The web UI at `/train` has a built-in voice test player — select a checkpoint, export it, type text, and listen. You can test checkpoints while training continues.

CLI alternative:

``` sh
echo "Hello, this is my custom voice!" | piper -m models/British_Narrator.0250.onnx --output_file test.wav
```

See [train/README.md](train/README.md) for the complete training guide.


## Multi-User Mode

``` sh
python3 -m piper_recording_studio --multi-user
```

Now a "login code" will be required to record. A directory `output/user_<code>/<language>` must exist for each user and language.
