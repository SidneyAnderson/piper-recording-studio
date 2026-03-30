import argparse
import asyncio
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import uuid4

import httpx
import hypercorn
from quart import (
    Quart,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from elevenlabs_generate.client import ElevenLabsConfig, synthesize, test_api_key, get_models, get_voice_info

_LOGGER = logging.getLogger(__name__)
_DIR = Path(__file__).parent


@dataclass
class Prompt:
    """Single prompt for the user to read."""

    group: str
    id: str
    text: str


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    #
    parser.add_argument(
        "--prompts",
        help="Path to prompts directory",
        action="append",
        default=[_DIR.parent / "prompts"],
    )
    parser.add_argument(
        "--output",
        help="Path to output directory",
        default=_DIR.parent / "output",
    )
    #
    parser.add_argument(
        "--multi-user",
        action="store_true",
        help="Require login code and user output directory to exist",
    )
    parser.add_argument("--cc0", action="store_true", help="Show public domain notice")
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    prompts_dirs = [Path(p) for p in args.prompts]
    output_dir = Path(args.output)
    css_dir = _DIR / "css"
    js_dir = _DIR / "js"
    img_dir = _DIR / "img"
    webfonts_dir = _DIR / "webfonts"

    prompts, languages = load_prompts(prompts_dirs)

    app = Quart("piper", template_folder=str(_DIR / "templates"))
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 mb
    app.secret_key = str(uuid4())

    @app.route("/")
    @app.route("/index.html")
    async def api_index() -> str:
        """Main page"""
        return await render_template(
            "index.html",
            languages=sorted(languages.items()),
            multi_user=args.multi_user,
            cc0=args.cc0,
        )

    @app.route("/done.html")
    async def api_done() -> str:
        return await render_template("done.html")

    @app.route("/record")
    async def api_record() -> str:
        """Record audio for a text prompt"""
        language = request.args["language"]

        if args.multi_user:
            user_id = request.args.get("userId")
            audio_dir = output_dir / f"user_{user_id}"
            user_dir = audio_dir / language
            if not user_dir.is_dir():
                _LOGGER.warning("No user/language directory: %s", user_dir)
                user_id = None
        else:
            user_id = None
            audio_dir = output_dir

        next_prompt, num_complete, num_items = get_next_prompt(
            prompts,
            audio_dir,
            language,
        )
        if next_prompt is None:
            return await render_template("done.html")

        complete_percent = 100 * (num_complete / num_items if num_items > 0 else 1)
        return await render_template(
            "record.html",
            language=language,
            prompt_group=next_prompt.group,
            prompt_id=next_prompt.id,
            text=next_prompt.text,
            num_complete=num_complete,
            num_items=num_items,
            complete_percent=complete_percent,
            multi_user=args.multi_user,
            user_id=user_id,
        )

    @app.route("/submit", methods=["POST"])
    async def api_submit() -> Response:
        """Submit audio for a text prompt"""
        form = await request.form
        language = form["language"]
        prompt_group = form["promptGroup"]
        prompt_id = form["promptId"]
        prompt_text = form["text"]
        audio_format = form["format"]

        files = await request.files
        assert "audio" in files, "No audio"

        suffix = ".webm"
        if "wav" in audio_format:
            suffix = ".wav"

        if args.multi_user:
            user_id = form["userId"]
            user_dir = output_dir / f"user_{user_id}"
            if not user_dir.is_dir():
                _LOGGER.warning("No user/language directory: %s", user_dir)
                raise ValueError("Invalid login code")

            audio_dir = user_dir
        else:
            audio_dir = output_dir

        # Save audio and transcription
        audio_path = audio_dir / language / prompt_group / f"{prompt_id}{suffix}"
        _LOGGER.debug("Saving to %s", audio_path)

        audio_path.parent.mkdir(parents=True, exist_ok=True)
        await files["audio"].save(audio_path)

        text_path = audio_path.parent / f"{prompt_id}.txt"
        text_path.write_text(prompt_text, encoding="utf-8")

        # Get next prompt
        next_prompt, num_complete, num_items = get_next_prompt(
            prompts,
            audio_dir,
            language,
        )
        if next_prompt is None:
            return jsonify({"done": True})

        complete_percent = 100 * (num_complete / num_items if num_items > 0 else 1)
        return jsonify(
            {
                "done": False,
                "promptGroup": next_prompt.group,
                "promptId": next_prompt.id,
                "promptText": next_prompt.text,
                "numComplete": num_complete,
                "numItems": num_items,
                "completePercent": complete_percent,
            }
        )

    @app.route("/upload")
    async def api_upload() -> str:
        """Upload an existing dataset"""
        language = request.args["language"]

        if args.multi_user:
            user_id = request.args.get("userId")
            audio_dir = output_dir / f"user_{user_id}"
            user_dir = audio_dir / language
            if not user_dir.is_dir():
                _LOGGER.warning("No user/language directory: %s", user_dir)
                raise RuntimeError("Invalid login code")
        else:
            user_id = None
            audio_dir = output_dir

        return await render_template(
            "upload.html",
            language=language,
            multi_user=args.multi_user,
            user_id=user_id,
        )

    @app.route("/dataset", methods=["POST"])
    async def api_dataset() -> str:
        """Upload an existing dataset"""
        form = await request.form
        language = form["language"]

        if args.multi_user:
            user_id = form.get("userId")
            audio_dir = output_dir / f"user_{user_id}"
            user_dir = audio_dir / language
            if not user_dir.is_dir():
                _LOGGER.warning("No user/language directory: %s", user_dir)
                raise RuntimeError("Invalid login code")
        else:
            user_id = None
            audio_dir = output_dir

        files = await request.files
        dataset_file = files["dataset"]
        upload_path = user_dir / "_uploads" / Path(dataset_file.filename).name
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        await dataset_file.save(upload_path)
        _LOGGER.debug("Saved dataset to %s", upload_path)

        return await render_template("done.html")

    @app.errorhandler(Exception)
    async def handle_error(err) -> Tuple[str, int]:
        """Return error as text."""
        _LOGGER.exception(err)
        return (f"{err.__class__.__name__}: {err}", 500)

    @app.route("/css/<path:filename>", methods=["GET"])
    async def css(filename) -> Response:
        """CSS static endpoint."""
        return await send_from_directory(css_dir, filename)

    @app.route("/js/<path:filename>", methods=["GET"])
    async def js(filename) -> Response:
        """Javascript static endpoint."""
        return await send_from_directory(js_dir, filename)

    @app.route("/img/<path:filename>", methods=["GET"])
    async def img(filename) -> Response:
        """Image static endpoint."""
        return await send_from_directory(img_dir, filename)

    @app.route("/webfonts/<path:filename>", methods=["GET"])
    async def webfonts(filename) -> Response:
        """Webfonts static endpoint."""
        return await send_from_directory(webfonts_dir, filename)

    # --- ElevenLabs generation routes ---

    @app.route("/generate")
    async def api_generate() -> str:
        """ElevenLabs TTS generation page"""
        return await render_template(
            "generate.html",
            languages=sorted(languages.items()),
        )

    @app.route("/train")
    async def api_train() -> str:
        """Training guide page"""
        # Count generated audio files per language
        num_samples = 0
        language = "en-GB"
        for lang_dir in output_dir.iterdir():
            if lang_dir.is_dir():
                count = len(list(lang_dir.rglob("*.wav")))
                if count > num_samples:
                    num_samples = count
                    language = lang_dir.name

        dataset_dir = output_dir.parent / f"dataset_{language}"
        training_dir = output_dir.parent / "training"
        batch_size = 32

        return await render_template(
            "train.html",
            language=language,
            num_samples=num_samples,
            dataset_dir=dataset_dir,
            training_dir=training_dir,
            batch_size=batch_size,
        )

    # --- Training process management ---
    training_process = None
    training_log_path = output_dir.parent / "training.log"
    piper_dir = output_dir.parent / "piper"
    training_dir = output_dir.parent / "training"
    checkpoints_dir = output_dir.parent / "checkpoints"

    @app.route("/api/training/status")
    async def api_training_status() -> Response:
        """Check dataset generation and export status."""
        nonlocal training_process
        generated = 0
        language = ""
        for lang_dir in output_dir.iterdir():
            if lang_dir.is_dir():
                count = len(list(lang_dir.rglob("*.wav")))
                if count > generated:
                    generated = count
                    language = lang_dir.name

        exported = 0
        export_dir = ""
        dataset_path = output_dir.parent / f"dataset_{language}"
        if dataset_path.exists():
            metadata = dataset_path / "metadata.csv"
            if metadata.exists():
                exported = sum(1 for _ in open(metadata))
                export_dir = str(dataset_path)

        preprocessed = (training_dir / "config.json").exists()
        checkpoint_exists = any(checkpoints_dir.glob("*.ckpt")) if checkpoints_dir.exists() else False

        training_running = training_process is not None and training_process.returncode is None

        return jsonify({
            "generated": generated,
            "language": language,
            "exported": exported,
            "export_dir": export_dir,
            "preprocessed": preprocessed,
            "checkpoint_exists": checkpoint_exists,
            "training_running": training_running,
        })

    @app.route("/api/training/start", methods=["POST"])
    async def api_training_start() -> Response:
        """Start training as a background process."""
        nonlocal training_process
        import subprocess

        if training_process is not None and training_process.returncode is None:
            return jsonify({"ok": False, "error": "Training is already running."})

        if not (training_dir / "config.json").exists():
            return jsonify({"ok": False, "error": "Dataset not preprocessed. Run the setup script first."})

        # Find checkpoint
        checkpoint_file = None
        if checkpoints_dir.exists():
            ckpts = sorted(checkpoints_dir.glob("*.ckpt"))
            if ckpts:
                checkpoint_file = str(ckpts[0])

        # Check for existing training checkpoints to resume from
        lightning_ckpt_dir = training_dir / "lightning_logs"
        resume_ckpt = None
        if lightning_ckpt_dir.exists():
            existing = sorted(lightning_ckpt_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
            if existing:
                resume_ckpt = str(existing[-1])

        data = await request.get_json()
        batch_size = data.get("batchSize", 48)
        max_epochs = data.get("maxEpochs", 1000)
        checkpoint_epochs = data.get("checkpointEpochs", 10)

        venv_python = piper_dir / "src" / "python" / ".venv" / "bin" / "python3"
        if not venv_python.exists():
            return jsonify({"ok": False, "error": f"Piper venv not found. Run: bash train/setup_training.sh"})

        cmd = [
            str(venv_python), "-m", "piper_train",
            "--dataset-dir", str(training_dir),
            "--accelerator", "gpu",
            "--devices", "1",
            "--batch-size", str(batch_size),
            "--validation-split", "0.0",
            "--num-test-examples", "0",
            "--max_epochs", str(max_epochs),
            "--checkpoint-epochs", str(checkpoint_epochs),
            "--precision", "32",
        ]

        if resume_ckpt:
            cmd.extend(["--resume_from_checkpoint", resume_ckpt])
        elif checkpoint_file:
            cmd.extend(["--resume_from_checkpoint", checkpoint_file])

        _LOGGER.info("Starting training: %s", " ".join(cmd))

        log_file = open(training_log_path, "w")
        training_process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(piper_dir / "src" / "python"),
        )

        return jsonify({"ok": True, "pid": training_process.pid})

    @app.route("/api/training/stop", methods=["POST"])
    async def api_training_stop() -> Response:
        """Stop the training process."""
        nonlocal training_process
        import signal

        if training_process is None or training_process.returncode is not None:
            return jsonify({"message": "Training is not running."})

        training_process.send_signal(signal.SIGINT)
        try:
            training_process.wait(timeout=30)
        except Exception:
            training_process.kill()

        return jsonify({"message": "Training stopped. Last checkpoint has been saved."})

    @app.route("/api/training/log")
    async def api_training_log() -> Response:
        """Stream training log file contents."""
        async def stream_log():
            if not training_log_path.exists():
                yield "Waiting for training to start...\n"
                return

            last_pos = 0
            while True:
                if training_log_path.exists():
                    with open(training_log_path, "r") as f:
                        f.seek(last_pos)
                        new_data = f.read()
                        if new_data:
                            last_pos = f.tell()
                            yield new_data

                # Check if training is still running
                if training_process is None or training_process.returncode is not None:
                    # Read any remaining data
                    if training_log_path.exists():
                        with open(training_log_path, "r") as f:
                            f.seek(last_pos)
                            remaining = f.read()
                            if remaining:
                                yield remaining
                    yield "\n[Training finished]\n"
                    return

                await asyncio.sleep(1)

        return Response(stream_log(), content_type="text/plain")

    @app.route("/api/training/checkpoints")
    async def api_training_checkpoints() -> Response:
        """List available training checkpoints."""
        checkpoints = []
        lightning_dir = training_dir / "lightning_logs"
        if lightning_dir.exists():
            ckpts = sorted(lightning_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
            checkpoints = [str(c.relative_to(training_dir)) for c in ckpts]

        onnx_path = output_dir.parent / "my_voice.onnx"
        return jsonify({
            "checkpoints": checkpoints,
            "onnx_exists": onnx_path.exists(),
        })

    @app.route("/api/training/export", methods=["POST"])
    async def api_training_export() -> Response:
        """Export a checkpoint to ONNX."""
        import subprocess

        data = await request.get_json()
        checkpoint = data.get("checkpoint", "")
        ckpt_path = training_dir / checkpoint

        if not ckpt_path.exists():
            return jsonify({"ok": False, "error": f"Checkpoint not found: {checkpoint}"})

        venv_python = piper_dir / "src" / "python" / ".venv" / "bin" / "python3"
        onnx_path = output_dir.parent / "my_voice.onnx"
        config_path = training_dir / "config.json"

        try:
            result = subprocess.run(
                [str(venv_python), "-m", "piper_train.export_onnx", str(ckpt_path), str(onnx_path)],
                capture_output=True, text=True, timeout=120,
                cwd=str(piper_dir / "src" / "python"),
            )
            if result.returncode != 0:
                return jsonify({"ok": False, "error": result.stderr[:500]})

            # Copy config alongside the model
            import shutil
            shutil.copy2(str(config_path), str(onnx_path) + ".json")

            return jsonify({"ok": True, "onnx_path": str(onnx_path)})
        except subprocess.TimeoutExpired:
            return jsonify({"ok": False, "error": "Export timed out."})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)})

    @app.route("/api/training/test-voice", methods=["POST"])
    async def api_training_test_voice() -> Response:
        """Generate speech with the exported ONNX model."""
        import subprocess

        data = await request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Text is required."}), 400

        onnx_path = output_dir.parent / "my_voice.onnx"
        if not onnx_path.exists():
            return jsonify({"error": "No exported model found. Export a checkpoint first."}), 404

        test_wav = output_dir.parent / "test_voice.wav"

        try:
            result = subprocess.run(
                ["piper", "-m", str(onnx_path), "--output_file", str(test_wav)],
                input=text, capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return jsonify({"error": result.stderr[:500]}), 500

            wav_bytes = test_wav.read_bytes()
            return Response(wav_bytes, content_type="audio/wav")
        except FileNotFoundError:
            return jsonify({"error": "piper not installed. Run: pip install piper-tts"}), 500
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Voice generation timed out."}), 500
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    env_path = output_dir.parent / ".env"

    def _load_env() -> dict:
        """Load key=value pairs from .env file."""
        config = {}
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    config[key.strip()] = value.strip()
        return config

    def _save_env(updates: dict) -> None:
        """Merge updates into .env file, preserving existing keys."""
        config = _load_env()
        config.update(updates)
        lines = [f"{k}={v}" for k, v in config.items()]
        env_path.write_text("\n".join(lines) + "\n")

    @app.route("/api/elevenlabs/config", methods=["GET"])
    async def api_elevenlabs_config() -> Response:
        """Return saved ElevenLabs config from .env."""
        config = _load_env()
        return jsonify({
            "apiKey": config.get("ELEVENLABS_API_KEY", ""),
            "voiceId": config.get("ELEVENLABS_VOICE_ID", ""),
            "modelId": config.get("ELEVENLABS_MODEL_ID", ""),
        })

    @app.route("/api/elevenlabs/models", methods=["POST"])
    async def api_elevenlabs_models() -> Response:
        """Fetch available TTS models from ElevenLabs."""
        data = await request.get_json()
        api_key = data.get("apiKey", "").strip()
        if not api_key:
            return jsonify({"error": "API key is required."}), 400
        try:
            async with httpx.AsyncClient() as client:
                models = await get_models(client, api_key)
            return jsonify({"models": models})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/elevenlabs/voice-info", methods=["POST"])
    async def api_elevenlabs_voice_info() -> Response:
        """Fetch voice details to determine compatible models."""
        data = await request.get_json()
        api_key = data.get("apiKey", "").strip()
        voice_id = data.get("voiceId", "").strip()
        if not api_key or not voice_id:
            return jsonify({"error": "API key and voice ID are required."}), 400
        try:
            async with httpx.AsyncClient() as client:
                info = await get_voice_info(client, api_key, voice_id)
            # Save validated voice ID to .env
            _save_env({"ELEVENLABS_VOICE_ID": voice_id})
            if info.get("model_ids"):
                _save_env({"ELEVENLABS_MODEL_ID": info["model_ids"][0]})
            return jsonify(info)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return jsonify({"error": "Voice not found. Check the voice ID."}), 404
            return jsonify({"error": f"API error: {exc.response.status_code}"}), 502
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/elevenlabs/test", methods=["POST"])
    async def api_elevenlabs_test() -> Response:
        """Test an ElevenLabs API key and save to .env on success."""
        data = await request.get_json()
        api_key = data.get("apiKey", "").strip()
        if not api_key:
            return jsonify({"ok": False, "error": "API key is required."}), 400
        try:
            async with httpx.AsyncClient() as client:
                info = await test_api_key(client, api_key)
            # Save config to .env on success
            env_updates = {"ELEVENLABS_API_KEY": api_key}
            voice_id = data.get("voiceId", "").strip()
            model_id = data.get("modelId", "").strip()
            if voice_id:
                env_updates["ELEVENLABS_VOICE_ID"] = voice_id
            if model_id:
                env_updates["ELEVENLABS_MODEL_ID"] = model_id
            _save_env(env_updates)
            return jsonify({"ok": True, **info})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                return jsonify({"ok": False, "error": "Invalid API key."}), 401
            return jsonify({"ok": False, "error": f"API error: {exc.response.status_code}"}), 502
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/elevenlabs/preview", methods=["POST"])
    async def api_elevenlabs_preview() -> Response:
        """Generate a single TTS preview and return WAV audio."""
        data = await request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Text is required."}), 400

        config = ElevenLabsConfig(
            api_key=data["apiKey"],
            voice_id=data["voiceId"],
            model_id=data["modelId"],
            sample_rate=int(data.get("sampleRate", 24000)),
            stability=float(data.get("stability", 0.5)),
            similarity_boost=float(data.get("similarityBoost", 0.75)),
        )
        try:
            async with httpx.AsyncClient() as client:
                wav_bytes = await synthesize(client, config, text)
            return Response(wav_bytes, content_type="audio/wav")
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", {})
                msg = detail.get("message", exc.response.text[:200]) if isinstance(detail, dict) else str(detail)[:200]
            except Exception:
                msg = exc.response.text[:200]
            return jsonify({"error": f"{exc.response.status_code}: {msg}"}), 502
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/elevenlabs/generate", methods=["POST"])
    async def api_elevenlabs_generate() -> Response:
        """Stream SSE progress while generating TTS audio via ElevenLabs."""
        data = await request.get_json()
        language = data["language"]
        config = ElevenLabsConfig(
            api_key=data["apiKey"],
            voice_id=data["voiceId"],
            model_id=data["modelId"],
            sample_rate=int(data.get("sampleRate", 24000)),
            stability=float(data.get("stability", 0.5)),
            similarity_boost=float(data.get("similarityBoost", 0.75)),
        )

        # Persist current config to .env
        _save_env({
            "ELEVENLABS_API_KEY": data["apiKey"],
            "ELEVENLABS_VOICE_ID": data["voiceId"],
            "ELEVENLABS_MODEL_ID": data["modelId"],
        })

        language_prompts = prompts.get(language, [])
        if not language_prompts:
            return Response(
                f"data: {json.dumps({'type': 'error', 'message': f'No prompts for language: {language}'})}\n\n",
                content_type="text/event-stream",
            )

        async def generate_stream():
            audio_dir = output_dir
            incomplete = []
            for prompt in language_prompts:
                text_path = audio_dir / language / prompt.group / f"{prompt.id}.txt"
                if not text_path.exists():
                    incomplete.append(prompt)

            total = len(language_prompts)
            already_done = total - len(incomplete)
            generated = 0
            failed = 0

            if not incomplete:
                yield f"data: {json.dumps({'type': 'done', 'generated': 0, 'failed': 0, 'total': total, 'message': 'All prompts already completed.'})}\n\n"
                return

            max_retries = 10
            async with httpx.AsyncClient() as client:
                for i, prompt in enumerate(incomplete):
                    current = already_done + generated + failed + 1
                    success = False

                    for attempt in range(1, max_retries + 1):
                        try:
                            wav_bytes = await synthesize(client, config, prompt.text)
                            audio_path = audio_dir / language / prompt.group / f"{prompt.id}.wav"
                            audio_path.parent.mkdir(parents=True, exist_ok=True)
                            audio_path.write_bytes(wav_bytes)

                            text_path = audio_path.parent / f"{prompt.id}.txt"
                            text_path.write_text(prompt.text, encoding="utf-8")

                            generated += 1
                            msg = f"[{current}/{total}] Saved {prompt.id}.wav ({len(wav_bytes)} bytes)"
                            yield f"data: {json.dumps({'type': 'progress', 'status': 'ok', 'message': msg, 'generated': already_done + generated, 'failed': failed, 'total': total})}\n\n"
                            success = True
                            break
                        except httpx.HTTPStatusError as exc:
                            try:
                                detail = exc.response.json().get("detail", {})
                                error_detail = detail.get("message", exc.response.text[:200]) if isinstance(detail, dict) else str(detail)[:200]
                            except Exception:
                                error_detail = exc.response.text[:200]
                            # Fatal errors — abort immediately
                            if exc.response.status_code in (400, 401, 403):
                                failed += 1
                                msg = f"[{current}/{total}] API error for {prompt.id}: {exc.response.status_code} — {error_detail}"
                                yield f"data: {json.dumps({'type': 'progress', 'status': 'error', 'message': msg, 'generated': already_done + generated, 'failed': failed, 'total': total})}\n\n"
                                yield f"data: {json.dumps({'type': 'error', 'message': f'Fatal error ({exc.response.status_code}): {error_detail} Aborting.'})}\n\n"
                                return
                            if exc.response.status_code == 429:
                                wait = 30
                                msg = f"[{current}/{total}] Rate limited. Waiting {wait}s... (attempt {attempt}/{max_retries})"
                                yield f"data: {json.dumps({'type': 'progress', 'status': 'error', 'message': msg, 'generated': already_done + generated, 'failed': failed, 'total': total})}\n\n"
                                await asyncio.sleep(wait)
                                continue
                            # Other HTTP errors — retry with backoff
                            wait = 5 * attempt
                            msg = f"[{current}/{total}] API error for {prompt.id}: {exc.response.status_code} — retrying in {wait}s (attempt {attempt}/{max_retries})"
                            yield f"data: {json.dumps({'type': 'progress', 'status': 'error', 'message': msg, 'generated': already_done + generated, 'failed': failed, 'total': total})}\n\n"
                            await asyncio.sleep(wait)
                        except Exception as exc:
                            # Network errors, timeouts — retry with backoff
                            wait = 5 * attempt
                            msg = f"[{current}/{total}] {type(exc).__name__} for {prompt.id} — retrying in {wait}s (attempt {attempt}/{max_retries})"
                            yield f"data: {json.dumps({'type': 'progress', 'status': 'error', 'message': msg, 'generated': already_done + generated, 'failed': failed, 'total': total})}\n\n"
                            await asyncio.sleep(wait)

                    if not success:
                        failed += 1
                        msg = f"[{current}/{total}] Failed {prompt.id} after {max_retries} attempts — skipping"
                        yield f"data: {json.dumps({'type': 'progress', 'status': 'error', 'message': msg, 'generated': already_done + generated, 'failed': failed, 'total': total})}\n\n"

                    if i < len(incomplete) - 1:
                        await asyncio.sleep(0.5)

            yield f"data: {json.dumps({'type': 'done', 'generated': already_done + generated, 'failed': failed, 'total': total})}\n\n"

        return Response(generate_stream(), content_type="text/event-stream")

    # Run web server
    hyp_config = hypercorn.config.Config()
    hyp_config.bind = [f"{args.host}:{args.port}"]

    asyncio.run(hypercorn.asyncio.serve(app, hyp_config))


# -----------------------------------------------------------------------------


def load_prompts(
    prompts_dirs: List[Path],
) -> Tuple[Dict[str, List[Prompt]], Dict[str, str]]:
    prompts = defaultdict(list)
    languages = {}

    for prompts_dir in prompts_dirs:
        for language_dir in prompts_dir.iterdir():
            if not language_dir.is_dir():
                continue

            name, code = language_dir.name.rsplit("_", maxsplit=1)
            languages[name] = code
            for prompt_path in language_dir.glob("*.txt"):
                _LOGGER.debug("Loading prompts from %s", prompt_path)
                prompt_group = prompt_path.stem
                with open(prompt_path, "r", encoding="utf-8") as prompt_file:
                    reader = csv.reader(prompt_file, delimiter="\t")
                    for i, row in enumerate(reader):
                        if len(row) == 1:
                            prompt_id = str(i)
                        else:
                            prompt_id = row[0]

                        prompts[code].append(
                            Prompt(group=prompt_group, id=prompt_id, text=row[-1])
                        )

    return prompts, languages


def get_next_prompt(
    prompts: Dict[str, List[Prompt]],
    output_dir: Path,
    language: str,
):
    language_prompts = prompts[language]
    language_dir = output_dir / language
    incomplete_prompts = []
    for prompt in language_prompts:
        text_path = language_dir / prompt.group / f"{prompt.id}.txt"
        if not text_path.exists():
            incomplete_prompts.append(prompt)

    num_items = len(language_prompts)
    num_complete = num_items - len(incomplete_prompts)

    if incomplete_prompts:
        next_prompt = incomplete_prompts[0]
    else:
        next_prompt = None

    return next_prompt, num_complete, num_items


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
