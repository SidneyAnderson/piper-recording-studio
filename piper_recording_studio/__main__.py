import argparse
import asyncio
import csv
import json
import logging
import os
import sys
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
from train.checkpoints import list_all as list_checkpoints, get_checkpoint_url, get_checkpoint_filename

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

    @app.errorhandler(404)
    async def handle_404(err) -> Tuple[str, int]:
        """Suppress noisy 404 logs for favicon, manifests, etc."""
        return ("Not Found", 404)

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

    @app.route("/api/training/export-dataset", methods=["POST"])
    async def api_training_export_dataset() -> Response:
        """Export audio to LJSpeech format."""
        import subprocess

        data = await request.get_json()
        language = data.get("language", "")
        if not language:
            return jsonify({"ok": False, "error": "Language is required."})

        input_path = output_dir / language
        if not input_path.exists():
            return jsonify({"ok": False, "error": f"No audio found for {language}."})

        dataset_path = output_dir.parent / f"dataset_{language}"

        async def stream():
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Exporting dataset...'})}\n\n"
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "export_dataset",
                     "--audio-glob", "*.wav",
                     str(input_path), str(dataset_path)],
                    capture_output=True, text=True, timeout=600,
                    cwd=str(output_dir.parent),
                )
                if result.returncode != 0:
                    yield f"data: {json.dumps({'type': 'error', 'message': result.stderr[:500]})}\n\n"
                    return

                count = 0
                metadata = dataset_path / "metadata.csv"
                if metadata.exists():
                    count = sum(1 for _ in open(metadata))

                yield f"data: {json.dumps({'type': 'done', 'message': f'Exported {count} samples to {dataset_path}'})}\n\n"
            except subprocess.TimeoutExpired:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Export timed out after 10 minutes.'})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

        return Response(stream(), content_type="text/event-stream")

    @app.route("/api/training/setup", methods=["POST"])
    async def api_training_setup() -> Response:
        """Run the training setup script (clone Piper, install deps)."""
        import subprocess

        setup_script = output_dir.parent / "setup_training.sh"
        if not setup_script.exists():
            return jsonify({"ok": False, "error": "Setup script not found."})

        async def stream():
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Running setup... This may take several minutes.'})}\n\n"
            try:
                process = subprocess.Popen(
                    ["bash", str(setup_script), "--skip-checkpoint"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=str(output_dir.parent),
                )
                for line in iter(process.stdout.readline, ""):
                    line = line.rstrip()
                    if line:
                        yield f"data: {json.dumps({'type': 'progress', 'message': line})}\n\n"
                process.wait()
                if process.returncode == 0:
                    yield f"data: {json.dumps({'type': 'done', 'message': 'Setup complete!'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Setup failed with exit code {process.returncode}'})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

        return Response(stream(), content_type="text/event-stream")

    @app.route("/api/training/preprocess", methods=["POST"])
    async def api_training_preprocess() -> Response:
        """Preprocess the exported dataset for training."""
        import subprocess

        data = await request.get_json()
        language = data.get("language", "")
        sample_rate = data.get("sampleRate", 22050)

        dataset_path = output_dir.parent / f"dataset_{language}"
        if not dataset_path.exists():
            return jsonify({"ok": False, "error": f"Dataset not exported yet for {language}."})

        venv_python = piper_dir / "src" / "python" / ".venv" / "bin" / "python3"
        if not venv_python.exists():
            return jsonify({"ok": False, "error": "Piper not installed. Run Setup first."})

        # Clear old training data if re-preprocessing
        if training_dir.exists():
            import shutil
            shutil.rmtree(training_dir)

        training_dir.mkdir(parents=True, exist_ok=True)

        # Extract language code (e.g., "en" from "en-GB")
        lang_code = language.split("-")[0] if "-" in language else language

        async def stream():
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Preprocessing dataset...'})}\n\n"
            try:
                process = subprocess.Popen(
                    [str(venv_python), "-m", "piper_train.preprocess",
                     "--language", lang_code,
                     "--input-dir", str(dataset_path),
                     "--output-dir", str(training_dir),
                     "--dataset-format", "ljspeech",
                     "--single-speaker",
                     "--sample-rate", str(sample_rate)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=str(piper_dir / "src" / "python"),
                )
                for line in iter(process.stdout.readline, ""):
                    line = line.rstrip()
                    if line:
                        yield f"data: {json.dumps({'type': 'progress', 'message': line})}\n\n"
                process.wait()
                if process.returncode == 0:
                    yield f"data: {json.dumps({'type': 'done', 'message': 'Preprocessing complete!'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Preprocessing failed with exit code {process.returncode}'})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

        return Response(stream(), content_type="text/event-stream")

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

        training_running = _is_training_running()

        # Check if piper is installed
        piper_installed = (piper_dir / "src" / "python" / ".venv" / "bin" / "python3").exists()

        return jsonify({
            "generated": generated,
            "language": language,
            "exported": exported,
            "export_dir": export_dir,
            "preprocessed": preprocessed,
            "checkpoint_exists": checkpoint_exists,
            "training_running": training_running,
            "piper_installed": piper_installed,
        })

    @app.route("/api/training/start", methods=["POST"])
    async def api_training_start() -> Response:
        """Start training as a background process."""
        nonlocal training_process
        import subprocess

        if _is_training_running():
            return jsonify({"ok": False, "error": "Training is already running."})

        if not (training_dir / "config.json").exists():
            return jsonify({"ok": False, "error": "Dataset not preprocessed. Run the setup script first."})

        # Find the downloaded pre-trained checkpoint (e.g. en_GB-cori-high.ckpt)
        # This is only used for the initial fine-tuning start
        checkpoint_file = None
        if checkpoints_dir.exists():
            ckpts = sorted(checkpoints_dir.glob("*.ckpt"))
            if ckpts:
                checkpoint_file = str(ckpts[0])

        # Check for existing training checkpoints to resume from
        # If found, these take priority over the downloaded checkpoint
        # since they contain the partially-trained model state
        lightning_ckpt_dir = training_dir / "lightning_logs"
        resume_ckpt = None
        if lightning_ckpt_dir.exists():
            existing = sorted(lightning_ckpt_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
            if existing:
                resume_ckpt = str(existing[-1])

        data = await request.get_json()
        batch_size = data.get("batchSize", 32)
        requested_epochs = data.get("maxEpochs", 1000)
        checkpoint_epochs = data.get("checkpointEpochs", 10)
        train_mode = data.get("trainMode", "finetune")

        venv_python = piper_dir / "src" / "python" / ".venv" / "bin" / "python3"
        if not venv_python.exists():
            return jsonify({"ok": False, "error": f"Piper venv not found. Run Setup in Step 2."})

        # Determine which checkpoint to resume from:
        # 1. If training was previously started, resume from latest training checkpoint
        # 2. If fine-tuning for the first time, use the downloaded pre-trained checkpoint
        # 3. If training from scratch, no checkpoint is used
        ckpt_to_use = None
        if resume_ckpt:
            ckpt_to_use = resume_ckpt
        elif train_mode == "finetune" and checkpoint_file:
            ckpt_to_use = checkpoint_file

        # Read checkpoint's current epoch using the training venv's python (has torch)
        max_epochs = requested_epochs
        if ckpt_to_use:
            try:
                result = subprocess.run(
                    [str(venv_python), "-c",
                     f"import torch,pathlib;torch.serialization.add_safe_globals([pathlib.PosixPath,pathlib.WindowsPath]);c=torch.load('{ckpt_to_use}',map_location='cpu',weights_only=False);print(c.get('epoch',0))"],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 0:
                    ckpt_epoch = int(result.stdout.strip())
                    max_epochs = ckpt_epoch + 1 + requested_epochs
                    _LOGGER.info("Checkpoint at epoch %d, will train %d additional epochs (max_epochs=%d)",
                                 ckpt_epoch + 1, requested_epochs, max_epochs)
                else:
                    _LOGGER.warning("Could not read checkpoint epoch: %s", result.stderr[:200])
            except Exception as exc:
                _LOGGER.warning("Could not read checkpoint epoch, using max_epochs=%d: %s", max_epochs, exc)

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

        if ckpt_to_use:
            cmd.extend(["--resume_from_checkpoint", ckpt_to_use])

        # Detect quality tier: check if any downloaded checkpoint is high quality,
        # or if the resume checkpoint was originally trained with high quality
        is_high = False
        if checkpoints_dir.exists():
            is_high = any("high" in f.name.lower() for f in checkpoints_dir.glob("*.ckpt"))
        if not is_high and ckpt_to_use:
            # Check file size: high quality checkpoints are ~950MB+, medium ~400MB
            try:
                ckpt_size_mb = Path(ckpt_to_use).stat().st_size / (1024 * 1024)
                if ckpt_size_mb > 800:
                    is_high = True
            except Exception:
                pass
        if is_high:
            cmd.extend(["--quality", "high"])
        # train_mode == "scratch" with no checkpoint: no checkpoint flag = from scratch

        _LOGGER.info("Starting training: %s", " ".join(cmd))

        log_file = open(training_log_path, "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        # start_new_session=True detaches training from the web server so
        # it survives server restarts (Ctrl+C won't kill training)
        training_process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(piper_dir / "src" / "python"),
            env=env,
            start_new_session=True,
        )

        # Save PID and start time so we can track training across server restarts
        import time as _time
        pid_path = output_dir.parent / "training.pid"
        pid_path.write_text(f"{training_process.pid}\n{_time.time()}")

        return jsonify({"ok": True, "pid": training_process.pid})

    def _read_training_pid_file() -> tuple:
        """Read PID and start time from training.pid file. Returns (pid, start_time) or (None, None)."""
        pid_path = output_dir.parent / "training.pid"
        if pid_path.exists():
            try:
                lines = pid_path.read_text().strip().split("\n")
                pid = int(lines[0])
                start_time = float(lines[1]) if len(lines) > 1 else 0
                os.kill(pid, 0)  # Check if process exists
                return pid, start_time
            except (ValueError, ProcessLookupError, PermissionError, IndexError):
                pid_path.unlink(missing_ok=True)
        return None, None

    def _is_training_running() -> bool:
        """Check if training is running, even across server restarts."""
        nonlocal training_process
        if training_process is not None and training_process.returncode is None:
            return True
        # Check PID file
        pid, _ = _read_training_pid_file()
        if pid is not None:
            return True
        # Check for any piper_train process (catches orphans)
        try:
            import subprocess
            result = subprocess.run(["pgrep", "-f", "piper_train"], capture_output=True)
            return result.returncode == 0
        except Exception:
            return False

    @app.route("/api/training/stop", methods=["POST"])
    async def api_training_stop() -> Response:
        """Stop all training processes."""
        nonlocal training_process
        import signal
        import subprocess

        pid_path = output_dir.parent / "training.pid"
        killed = []

        # Find ALL piper_train processes (handles duplicates and orphans)
        try:
            result = subprocess.run(
                ["pgrep", "-f", "piper_train"],
                capture_output=True, text=True,
            )
            pids = [int(p.strip()) for p in result.stdout.strip().split("\n") if p.strip()]
        except Exception:
            pids = []

        # Also check PID file
        pid_from_file, _ = _read_training_pid_file()
        if pid_from_file and pid_from_file not in pids:
            pids.append(pid_from_file)

        if not pids:
            return jsonify({"message": "Training is not running."})

        for pid in pids:
            try:
                # Try process group kill first (graceful), then direct
                try:
                    os.killpg(os.getpgid(pid), signal.SIGINT)
                except (ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGINT)
                killed.append(pid)
            except (ProcessLookupError, PermissionError):
                pass

        # Wait for processes to exit
        import time as _time
        for _ in range(15):
            _time.sleep(1)
            still_running = False
            for pid in killed:
                try:
                    os.kill(pid, 0)
                    still_running = True
                except ProcessLookupError:
                    pass
            if not still_running:
                break

        # Force kill any stragglers
        for pid in killed:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        # Clean up
        if training_process is not None:
            try:
                training_process.wait(timeout=5)
            except Exception:
                pass
            training_process = None
        pid_path.unlink(missing_ok=True)

        return jsonify({"message": f"Training stopped. Killed {len(killed)} process(es)."})

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
                if not _is_training_running():
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

    @app.route("/api/training/progress")
    async def api_training_progress() -> Response:
        """Return current training progress by scanning checkpoints.

        Since Piper doesn't log per-epoch output, we determine progress by:
        1. Scanning checkpoint files for the latest epoch number
        2. Estimating additional epochs based on time since last checkpoint
        3. Calculating start_epoch from the first checkpoint minus save interval

        The frontend shows user-friendly epoch counts (1 to N) by subtracting
        start_epoch from the current epoch.
        """
        import re
        nonlocal training_process

        running = _is_training_running()

        # Find all training checkpoints and determine progress
        lightning_dir = training_dir / "lightning_logs"
        latest_epoch = None
        first_epoch = None
        total_checkpoints = 0

        if lightning_dir.exists():
            ckpts = sorted(lightning_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
            total_checkpoints = len(ckpts)
            for ckpt in ckpts:
                m = re.search(r'epoch=(\d+)', ckpt.name)
                if m:
                    epoch = int(m.group(1))
                    if first_epoch is None:
                        first_epoch = epoch
                    latest_epoch = epoch

        # Estimate the base epoch (where fine-tuning started)
        # First training checkpoint is saved checkpoint_interval epochs after the base
        checkpoint_interval = 10  # default
        if first_epoch is not None:
            start_epoch = first_epoch - checkpoint_interval
        else:
            start_epoch = 0

        # Also estimate current epoch from events file modification time
        # Each epoch takes ~7s, so we can estimate epochs since last checkpoint
        estimated_epoch = latest_epoch
        if latest_epoch is not None and running:
            # Find the latest events file
            events_files = sorted(lightning_dir.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
            if events_files:
                latest_events_mtime = events_files[-1].stat().st_mtime
                # Find the latest checkpoint mtime
                latest_ckpt_files = sorted(lightning_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
                if latest_ckpt_files:
                    latest_ckpt_mtime = latest_ckpt_files[-1].stat().st_mtime
                    # If events file is newer than last checkpoint, estimate additional epochs
                    if latest_events_mtime > latest_ckpt_mtime:
                        import time
                        seconds_since_ckpt = time.time() - latest_ckpt_mtime
                        # Rough estimate: ~7 seconds per epoch for high quality
                        estimated_extra = int(seconds_since_ckpt / 7)
                        estimated_epoch = latest_epoch + estimated_extra

        # Read last few lines of log for recent activity
        last_log_lines = []
        if training_log_path.exists():
            try:
                with open(training_log_path, "r") as f:
                    lines = f.readlines()
                    last_log_lines = [l.rstrip() for l in lines[-5:] if l.strip()]
            except Exception:
                pass

        # Get training start time from PID file
        _, train_start_time = _read_training_pid_file()

        return jsonify({
            "running": running,
            "latest_epoch": latest_epoch,
            "estimated_epoch": estimated_epoch,
            "start_epoch": start_epoch,
            "total_checkpoints": total_checkpoints,
            "last_log": last_log_lines,
            "start_time": train_start_time or 0,
        })

    @app.route("/api/training/stats")
    async def api_training_stats() -> Response:
        """Read loss values from TensorBoard events files."""
        lightning_dir = training_dir / "lightning_logs"
        loss_gen = []
        loss_disc = []

        if not lightning_dir.exists():
            return jsonify({"loss_gen": [], "loss_disc": []})

        try:
            # Find all events files across versions
            events_files = sorted(lightning_dir.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
            if not events_files:
                return jsonify({"loss_gen": [], "loss_disc": []})

            # Use the training venv's python to read TensorBoard events
            venv_python = piper_dir / "src" / "python" / ".venv" / "bin" / "python3"
            if not venv_python.exists():
                return jsonify({"loss_gen": [], "loss_disc": []})

            import subprocess
            # Read all events files
            events_paths = [str(f) for f in events_files]
            script = """
import json, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
results = {"loss_gen": [], "loss_disc": []}
for path in sys.argv[1:]:
    try:
        ea = EventAccumulator(path)
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        if 'loss_gen_all' in tags:
            for s in ea.Scalars('loss_gen_all'):
                results["loss_gen"].append({"step": s.step, "value": round(s.value, 4)})
        if 'loss_disc_all' in tags:
            for s in ea.Scalars('loss_disc_all'):
                results["loss_disc"].append({"step": s.step, "value": round(s.value, 4)})
    except Exception:
        pass
print(json.dumps(results))
"""
            result = subprocess.run(
                [str(venv_python), "-c", script] + events_paths,
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout.strip())
                return jsonify(data)
        except Exception as exc:
            _LOGGER.warning("Could not read training stats: %s", exc)

        return jsonify({"loss_gen": [], "loss_disc": []})

    @app.route("/api/training/checkpoints")
    async def api_training_checkpoints() -> Response:
        """List available training checkpoints with epoch info.

        Returns both internal epoch numbers and a base_epoch offset so the
        frontend can calculate user-facing epoch numbers (1 to N).
        """
        import re
        checkpoints = []
        first_epoch = None
        checkpoint_interval = 10

        lightning_dir = training_dir / "lightning_logs"
        if lightning_dir.exists():
            ckpts = sorted(lightning_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
            for ckpt in ckpts:
                m = re.search(r'epoch=(\d+)', ckpt.name)
                epoch = int(m.group(1)) if m else None
                if epoch is not None and first_epoch is None:
                    first_epoch = epoch
                checkpoints.append({
                    "path": str(ckpt.relative_to(training_dir)),
                    "epoch": epoch,
                    "name": ckpt.name,
                    "size_mb": round(ckpt.stat().st_size / (1024 * 1024)),
                })

        # base_epoch is where fine-tuning started (first checkpoint minus save interval)
        base_epoch = (first_epoch - checkpoint_interval) if first_epoch is not None else 0

        # Check if any exported models exist
        has_models = models_dir.exists() and any(models_dir.glob("*.onnx"))
        return jsonify({
            "checkpoints": checkpoints,
            "base_epoch": base_epoch,
            "onnx_exists": has_models,
        })

    models_dir = output_dir.parent / "models"

    @app.route("/api/training/export", methods=["POST"])
    async def api_training_export() -> Response:
        """Export a checkpoint to ONNX with profile-based naming."""
        import re
        import subprocess
        import shutil

        data = await request.get_json()
        checkpoint = data.get("checkpoint", "")
        profile_name = data.get("profileName", "voice")
        user_epoch = data.get("userEpoch", None)  # User-facing epoch number (1 to N)
        ckpt_path = training_dir / checkpoint

        if not ckpt_path.exists():
            return jsonify({"ok": False, "error": f"Checkpoint not found: {checkpoint}"})

        # Use user-facing epoch if provided, otherwise fall back to internal
        if user_epoch is not None:
            epoch = int(user_epoch)
        else:
            m = re.search(r'epoch=(\d+)', ckpt_path.name)
            epoch = int(m.group(1)) + 1 if m else 0

        # Sanitize profile name
        safe_name = "".join(c for c in profile_name if c.isalnum() or c in " _-").strip().replace(" ", "_")
        if not safe_name:
            safe_name = "voice"

        models_dir.mkdir(parents=True, exist_ok=True)
        onnx_filename = f"{safe_name}.{epoch}.onnx"
        onnx_path = models_dir / onnx_filename
        config_path = training_dir / "config.json"

        venv_python = piper_dir / "src" / "python" / ".venv" / "bin" / "python3"

        try:
            result = subprocess.run(
                [str(venv_python), "-m", "piper_train.export_onnx", str(ckpt_path), str(onnx_path)],
                capture_output=True, text=True, timeout=120,
                cwd=str(piper_dir / "src" / "python"),
            )
            if result.returncode != 0:
                return jsonify({"ok": False, "error": result.stderr[:500]})

            # Copy config alongside the model
            shutil.copy2(str(config_path), str(onnx_path) + ".json")

            return jsonify({
                "ok": True,
                "onnx_path": str(onnx_path),
                "onnx_name": onnx_filename,
                "epoch": epoch,
            })
        except subprocess.TimeoutExpired:
            return jsonify({"ok": False, "error": "Export timed out."})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)})

    @app.route("/api/training/models")
    async def api_training_models() -> Response:
        """List exported ONNX models."""
        model_list = []
        if models_dir.exists():
            for f in sorted(models_dir.glob("*.onnx")):
                model_list.append({
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / (1024 * 1024)),
                })
        return jsonify({"models": model_list})

    @app.route("/api/training/test-voice", methods=["POST"])
    async def api_training_test_voice() -> Response:
        """Generate speech with an exported ONNX model."""
        import subprocess

        data = await request.get_json()
        text = data.get("text", "").strip()
        model_name = data.get("model", "")
        if not text:
            return jsonify({"error": "Text is required."}), 400

        # Find the model
        if model_name:
            onnx_path = models_dir / model_name
        else:
            # Fall back to latest model
            if models_dir.exists():
                models = sorted(models_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
                onnx_path = models[-1] if models else None
            else:
                onnx_path = None

        if not onnx_path or not onnx_path.exists():
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

    @app.route("/api/training/available-checkpoints")
    async def api_available_checkpoints() -> Response:
        """List all available pre-trained checkpoints."""
        return jsonify({"checkpoints": list_checkpoints()})

    @app.route("/api/training/download-checkpoint", methods=["POST"])
    async def api_download_checkpoint() -> Response:
        """Download a pre-trained checkpoint from HuggingFace."""
        import subprocess

        data = await request.get_json()
        locale = data.get("locale", "")
        voice = data.get("voice", "")
        quality = data.get("quality", "")

        url = get_checkpoint_url(locale, voice, quality)
        if not url:
            return jsonify({"ok": False, "error": "Checkpoint not found in catalog."})

        filename = get_checkpoint_filename(locale, voice, quality)
        dest = checkpoints_dir / filename
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            return jsonify({"ok": True, "message": "Checkpoint already downloaded.", "path": str(dest)})

        async def download_stream():
            yield f"data: {json.dumps({'type': 'progress', 'message': f'Downloading {filename}...'})}\n\n"
            try:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    async with client.stream("GET", url, timeout=600.0) as resp:
                        resp.raise_for_status()
                        total = int(resp.headers.get("content-length", 0))
                        downloaded = 0
                        with open(dest, "wb") as f:
                            async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total > 0:
                                    pct = int(downloaded / total * 100)
                                    mb = downloaded // (1024 * 1024)
                                    total_mb = total // (1024 * 1024)
                                    yield f"data: {json.dumps({'type': 'progress', 'message': f'Downloading... {mb}/{total_mb} MB ({pct}%)', 'percent': pct})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'message': f'Downloaded {filename}', 'path': str(dest)})}\n\n"
            except Exception as exc:
                if dest.exists():
                    dest.unlink()
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

        return Response(download_stream(), content_type="text/event-stream")

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

    # --- Voice profiles ---
    profiles_dir = output_dir.parent / "voice_profiles"

    @app.route("/api/profiles", methods=["GET"])
    async def api_profiles_list() -> Response:
        """List saved voice profiles with training stats."""
        profiles = []
        if profiles_dir.exists():
            for f in sorted(profiles_dir.glob("*.json")):
                try:
                    data = json.loads(f.read_text())
                    profile_id = f.stem
                    data["id"] = profile_id

                    # Count generated audio files
                    lang = data.get("language", "")
                    profile_output = output_dir.parent / "voice_data" / profile_id / "output"
                    if profile_output.exists() and lang:
                        lang_dir = profile_output / lang
                        data["generated"] = len(list(lang_dir.rglob("*.wav"))) if lang_dir.exists() else 0
                    else:
                        # Check legacy output dir
                        if lang:
                            lang_dir = output_dir / lang
                            data["generated"] = len(list(lang_dir.rglob("*.wav"))) if lang_dir.exists() else 0
                        else:
                            data["generated"] = 0

                    # Check training state
                    profile_training = output_dir.parent / "voice_data" / profile_id / "training"
                    data["has_training"] = profile_training.exists() and (profile_training / "config.json").exists()

                    # Check for exported models matching this profile name
                    safe_id = profile_id.replace(" ", "_")
                    data["has_model"] = models_dir.exists() and any(
                        f.name.startswith(safe_id + ".") for f in models_dir.glob("*.onnx")
                    ) if models_dir.exists() else False

                    profiles.append(data)
                except Exception:
                    pass
        return jsonify({"profiles": profiles})

    @app.route("/api/profiles", methods=["POST"])
    async def api_profiles_save() -> Response:
        """Save or update a voice profile."""
        data = await request.get_json()
        name = data.get("name", "").strip()
        if not name:
            return jsonify({"ok": False, "error": "Profile name is required."})

        profiles_dir.mkdir(parents=True, exist_ok=True)

        # Use name as filename (sanitized)
        safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip().replace(" ", "_")
        profile_path = profiles_dir / f"{safe_name}.json"

        profile = {
            "name": name,
            "voice_id": data.get("voiceId", ""),
            "model_id": data.get("modelId", ""),
            "language": data.get("language", ""),
            "training_mode": data.get("trainingMode", "finetune"),
        }
        profile_path.write_text(json.dumps(profile, indent=2))
        return jsonify({"ok": True, "id": safe_name})

    @app.route("/api/profiles/<profile_id>", methods=["DELETE"])
    async def api_profiles_delete(profile_id: str) -> Response:
        """Delete a voice profile and optionally its data."""
        import shutil

        data = await request.get_json() if request.content_length else {}
        delete_data = data.get("deleteData", False) if data else False

        profile_path = profiles_dir / f"{profile_id}.json"
        if not profile_path.exists():
            return jsonify({"ok": False, "error": "Profile not found."})

        deleted_items = []

        if delete_data:
            # Remove profile-specific voice_data directory
            profile_data_dir = output_dir.parent / "voice_data" / profile_id
            if profile_data_dir.exists():
                shutil.rmtree(profile_data_dir)
                deleted_items.append("voice data")

            # Remove exported models matching this profile name
            if models_dir.exists():
                safe_id = profile_id.replace(" ", "_")
                for model_file in models_dir.glob(f"{safe_id}.*"):
                    model_file.unlink()
                    deleted_items.append(f"model: {model_file.name}")

            # Remove training checkpoints and logs
            if training_dir.exists():
                lightning_dir = training_dir / "lightning_logs"
                if lightning_dir.exists():
                    shutil.rmtree(lightning_dir)
                    deleted_items.append("training checkpoints")

        profile_path.unlink()
        return jsonify({"ok": True, "deleted": deleted_items})

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
