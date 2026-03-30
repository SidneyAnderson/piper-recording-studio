"""Async ElevenLabs TTS client used by both CLI and web UI."""

import io
import logging
import struct
import wave
from dataclasses import dataclass
from typing import Optional

import httpx

_LOGGER = logging.getLogger(__name__)

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_MODEL_ID = "eleven_monolingual_v1"


@dataclass
class ElevenLabsConfig:
    api_key: str
    voice_id: str
    model_id: str = DEFAULT_MODEL_ID
    sample_rate: int = DEFAULT_SAMPLE_RATE
    stability: float = 0.5
    similarity_boost: float = 0.75


def _pcm_to_wav(pcm_data: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM 16-bit mono data in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


async def test_api_key(
    client: httpx.AsyncClient,
    api_key: str,
) -> dict:
    """Validate an API key by hitting /v1/user. Returns user info or raises."""
    url = f"{ELEVENLABS_BASE_URL}/user"
    headers = {"xi-api-key": api_key}
    response = await client.get(url, headers=headers, timeout=15.0)
    response.raise_for_status()
    data = response.json()
    subscription = data.get("subscription", {})
    return {
        "character_count": subscription.get("character_count", 0),
        "character_limit": subscription.get("character_limit", 0),
        "tier": subscription.get("tier", "unknown"),
    }


async def get_models(
    client: httpx.AsyncClient,
    api_key: str,
) -> list[dict]:
    """Fetch available models from /v1/models."""
    url = f"{ELEVENLABS_BASE_URL}/models"
    headers = {"xi-api-key": api_key}
    response = await client.get(url, headers=headers, timeout=15.0)
    response.raise_for_status()
    models = response.json()
    return [
        {
            "model_id": m["model_id"],
            "name": m.get("name", m["model_id"]),
            "can_do_text_to_speech": m.get("can_do_text_to_speech", False),
            "languages": [lang.get("language_id", "") for lang in m.get("languages", [])],
        }
        for m in models
        if m.get("can_do_text_to_speech", False)
    ]


async def get_voice_info(
    client: httpx.AsyncClient,
    api_key: str,
    voice_id: str,
) -> dict:
    """Fetch voice details including high_quality_base_model_ids."""
    url = f"{ELEVENLABS_BASE_URL}/voices/{voice_id}"
    headers = {"xi-api-key": api_key}
    response = await client.get(url, headers=headers, timeout=15.0)
    response.raise_for_status()
    data = response.json()
    return {
        "name": data.get("name", ""),
        "model_ids": data.get("high_quality_base_model_ids", []),
    }


async def synthesize(
    client: httpx.AsyncClient,
    config: ElevenLabsConfig,
    text: str,
) -> bytes:
    """Send text to ElevenLabs TTS and return WAV audio bytes."""
    url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{config.voice_id}"
    params = {"output_format": f"pcm_{config.sample_rate}"}
    headers = {
        "xi-api-key": config.api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": config.model_id,
        "voice_settings": {
            "stability": config.stability,
            "similarity_boost": config.similarity_boost,
        },
    }

    response = await client.post(
        url, json=payload, params=params, headers=headers, timeout=120.0
    )
    response.raise_for_status()

    pcm_data = response.content
    return _pcm_to_wav(pcm_data, config.sample_rate)
