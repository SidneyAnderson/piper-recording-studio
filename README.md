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

Install ffmpeg:

``` sh
sudo apt-get install ffmpeg
```

Install exporting dependencies:

``` sh
python3 -m pip install -r requirements_export.txt
```

Export recordings for a language to a Piper-compatible dataset (LJSpeech format):

``` sh
python3 -m export_dataset output/<language>/ /path/to/dataset
```

Requires a non-Docker install. If you used Docker to record your dataset, you may need to adjust the permissions of the output directory:

``` sh
sudo chown -R "$(id -u):$(id -u)" output/
```

See `--help` for more options. You may need to adjust the silence detection parameters to correctly remove button clicks and keypresses.


## ElevenLabs TTS Generation

Generate audio from prompts using the [ElevenLabs](https://elevenlabs.io/) API instead of recording manually. Useful for creating synthetic training datasets.

### Web UI

Start the recording studio as normal, then click **ElevenLabs TTS** on the home page (or visit http://localhost:8000/generate). Enter your API key, voice ID, model ID, select a language, and click Start. Progress streams in real time.

### CLI

``` sh
python3 -m elevenlabs_generate \
  --api-key YOUR_API_KEY \
  --voice-id YOUR_VOICE_ID \
  --model-id eleven_monolingual_v1 \
  --language en-US
```

The API key can also be set via the `ELEVENLABS_API_KEY` environment variable.

Additional options:

| Flag | Default | Description |
|------|---------|-------------|
| `--prompts` | `prompts/` | Path to prompts directory |
| `--output` | `output/` | Path to output directory |
| `--sample-rate` | `24000` | Audio sample rate |
| `--stability` | `0.5` | Voice stability (0.0–1.0) |
| `--similarity-boost` | `0.75` | Voice similarity boost (0.0–1.0) |
| `--rate-limit-delay` | `0.5` | Seconds between API calls |

Generation is **resumable** — it skips prompts that already have output files. Audio is saved as WAV in the same `output/<language>/` structure used by the recorder, so the existing export pipeline works without changes.


## Multi-User Mode

``` sh
python3 -m piper_recording_studio --multi-user
```

Now a "login code" will be required to record. A directory `output/user_<code>/<language>` must exist for each user and language.
