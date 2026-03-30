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
