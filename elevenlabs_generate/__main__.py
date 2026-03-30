"""CLI tool to batch-generate TTS audio via ElevenLabs for Piper Recording Studio prompts."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import httpx

# Reuse the prompt-loading logic from the main app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_recording_studio.__main__ import Prompt, load_prompts

from .client import DEFAULT_MODEL_ID, DEFAULT_SAMPLE_RATE, ElevenLabsConfig, synthesize, test_api_key

_LOGGER = logging.getLogger(__name__)


def get_incomplete_prompts(
    prompts: list[Prompt],
    output_dir: Path,
    language: str,
) -> list[Prompt]:
    """Return prompts that have not yet been generated."""
    language_dir = output_dir / language
    incomplete = []
    for prompt in prompts:
        text_path = language_dir / prompt.group / f"{prompt.id}.txt"
        if not text_path.exists():
            incomplete.append(prompt)
    return incomplete


async def generate_all(
    config: ElevenLabsConfig,
    prompts: list[Prompt],
    output_dir: Path,
    language: str,
    rate_limit_delay: float = 0.5,
) -> dict:
    """Generate TTS audio for all incomplete prompts. Returns stats dict."""
    incomplete = get_incomplete_prompts(prompts, output_dir, language)
    total = len(prompts)
    already_done = total - len(incomplete)

    if not incomplete:
        _LOGGER.info("All %d prompts already completed for %s", total, language)
        return {"total": total, "generated": 0, "failed": 0, "skipped": already_done}

    _LOGGER.info(
        "%d/%d prompts remaining for %s", len(incomplete), total, language
    )

    generated = 0
    failed = 0

    async with httpx.AsyncClient() as client:
        for i, prompt in enumerate(incomplete):
            current = already_done + generated + failed + 1
            _LOGGER.info(
                "[%d/%d] Generating: %s (id=%s)", current, total, prompt.text[:60], prompt.id
            )

            audio_path = output_dir / language / prompt.group / f"{prompt.id}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                wav_bytes = await synthesize(client, config, prompt.text)
                audio_path.write_bytes(wav_bytes)

                text_path = audio_path.parent / f"{prompt.id}.txt"
                text_path.write_text(prompt.text, encoding="utf-8")

                generated += 1
                _LOGGER.info(
                    "[%d/%d] Saved %s (%d bytes)", current, total, audio_path.name, len(wav_bytes)
                )
            except httpx.HTTPStatusError as exc:
                failed += 1
                _LOGGER.error(
                    "[%d/%d] API error for prompt %s: %s %s",
                    current, total, prompt.id, exc.response.status_code, exc.response.text[:200],
                )
                if exc.response.status_code == 401:
                    _LOGGER.error("Invalid API key. Aborting.")
                    break
                if exc.response.status_code == 429:
                    _LOGGER.warning("Rate limited. Waiting 30s before retry...")
                    await asyncio.sleep(30)
                    # Retry this prompt once
                    try:
                        wav_bytes = await synthesize(client, config, prompt.text)
                        audio_path.write_bytes(wav_bytes)
                        text_path = audio_path.parent / f"{prompt.id}.txt"
                        text_path.write_text(prompt.text, encoding="utf-8")
                        generated += 1
                        failed -= 1
                        _LOGGER.info("[%d/%d] Retry succeeded", current, total)
                    except Exception:
                        _LOGGER.error("[%d/%d] Retry also failed", current, total)
            except Exception:
                failed += 1
                _LOGGER.exception("[%d/%d] Unexpected error for prompt %s", current, total, prompt.id)

            # Small delay to avoid hammering the API
            if i < len(incomplete) - 1:
                await asyncio.sleep(rate_limit_delay)

    stats = {"total": total, "generated": generated, "failed": failed, "skipped": already_done}
    _LOGGER.info("Done. Generated=%d, Failed=%d, Skipped=%d, Total=%d",
                 generated, failed, already_done, total)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TTS audio from ElevenLabs for Piper Recording Studio prompts"
    )
    parser.add_argument("--api-key", default=os.environ.get("ELEVENLABS_API_KEY"),
                        help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var)")
    parser.add_argument("--voice-id", default=None, help="ElevenLabs voice ID")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="ElevenLabs model ID")
    parser.add_argument("--language", default=None,
                        help="Language code to generate (e.g. en-US)")
    parser.add_argument("--test", action="store_true",
                        help="Test the API key and exit")
    parser.add_argument("--prompts", default=str(Path(__file__).resolve().parent.parent / "prompts"),
                        help="Path to prompts directory")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent.parent / "output"),
                        help="Path to output directory")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                        help="Audio sample rate (default: 24000)")
    parser.add_argument("--stability", type=float, default=0.5,
                        help="Voice stability (0.0-1.0, default: 0.5)")
    parser.add_argument("--similarity-boost", type=float, default=0.75,
                        help="Voice similarity boost (0.0-1.0, default: 0.75)")
    parser.add_argument("--rate-limit-delay", type=float, default=0.5,
                        help="Seconds between API calls (default: 0.5)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.api_key:
        _LOGGER.error("API key required. Use --api-key or set ELEVENLABS_API_KEY env var.")
        sys.exit(1)

    if args.test:
        async def _test():
            async with httpx.AsyncClient() as client:
                info = await test_api_key(client, args.api_key)
            print(f"API key is valid!")
            print(f"  Tier:       {info['tier']}")
            print(f"  Characters: {info['character_count']:,} / {info['character_limit']:,}")
        try:
            asyncio.run(_test())
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                _LOGGER.error("Invalid API key.")
            else:
                _LOGGER.error("API error: %s", exc.response.status_code)
            sys.exit(1)
        except Exception as exc:
            _LOGGER.error("Connection error: %s", exc)
            sys.exit(1)
        sys.exit(0)

    if not args.voice_id:
        _LOGGER.error("--voice-id is required for generation.")
        sys.exit(1)
    if not args.language:
        _LOGGER.error("--language is required for generation.")
        sys.exit(1)

    prompts_dir = Path(args.prompts)
    output_dir = Path(args.output)

    all_prompts, languages = load_prompts([prompts_dir])
    if args.language not in all_prompts:
        available = ", ".join(sorted(all_prompts.keys()))
        _LOGGER.error("Language '%s' not found. Available: %s", args.language, available)
        sys.exit(1)

    config = ElevenLabsConfig(
        api_key=args.api_key,
        voice_id=args.voice_id,
        model_id=args.model_id,
        sample_rate=args.sample_rate,
        stability=args.stability,
        similarity_boost=args.similarity_boost,
    )

    language_prompts = all_prompts[args.language]
    _LOGGER.info("Starting ElevenLabs generation for %s (%d prompts)", args.language, len(language_prompts))

    stats = asyncio.run(generate_all(config, language_prompts, output_dir, args.language, args.rate_limit_delay))

    if stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
