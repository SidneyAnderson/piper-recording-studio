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


### Building

``` sh
docker build . -t rhasspy/piper-recording-studio
```


## Installing without Docker

``` sh
git clone https://github.com/rhasspy/piper-recording-studio.git
cd piper-recording-studio/

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

System dependencies:

``` sh
sudo apt-get install ffmpeg espeak-ng
```


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

Note: the 44100 Hz sample rate requires an ElevenLabs Pro tier or above.

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
| `--sample-rate` | `24000` | Audio sample rate (44100 requires Pro tier) |
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

### Quick start

``` sh
# 1. Export dataset (--audio-glob '*.wav' required for ElevenLabs audio)
python3 -m export_dataset --audio-glob '*.wav' output/en-GB/ dataset_en-GB/

# 2. Run the automated setup (clones Piper, downloads checkpoint, preprocesses)
bash train/setup_training.sh

# 3. Follow the printed instructions to start training
```

The setup script handles:
- Cloning the Piper repo and installing dependencies (with pinned pip<24.1 for compatibility)
- Downloading the English medium pre-trained checkpoint (~400 MB)
- Preprocessing your dataset into training-ready tensors

### Web UI guide

Visit http://localhost:8000/train (or click **Training Guide** on the home page) for a step-by-step walkthrough with commands tailored to your dataset, including GPU batch size recommendations and monitoring tips.

### Selecting a checkpoint

The web UI at `/train` includes a **checkpoint browser** with 30+ languages, multiple voices, and low/medium/high quality tiers. Select your language, voice, and quality — then click Download. The high quality tier (~1 GB) is recommended when your GPU has enough VRAM.

Fine-tuning from a pre-trained checkpoint produces significantly better results than training from scratch with fewer than 5,000 samples. The checkpoint provides existing knowledge of speech patterns — training only adapts the voice characteristics.

All checkpoints sourced from: https://huggingface.co/datasets/rhasspy/piper-checkpoints

### Batch size by GPU VRAM

| VRAM | Batch Size |
|------|-----------|
| 8 GB | 12 |
| 10-12 GB | 24 |
| 16 GB | 32 |
| 24+ GB | 32-64 |

If you get out-of-memory errors during training, reduce the batch size. Larger batch sizes train faster but don't affect final model quality.

See [train/README.md](train/README.md) for the complete training guide.


## Multi-User Mode

``` sh
python3 -m piper_recording_studio --multi-user
```

Now a "login code" will be required to record. A directory `output/user_<code>/<language>` must exist for each user and language.
