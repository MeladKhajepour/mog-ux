"""Tests for modulate.py — Modulate Velma 2 API integration."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure env is loaded before importing module
os.environ.setdefault("MODULATE_API_KEY", "test-key")

from modulate import analyze_full_audio, _map_emotion


# --- Unit tests for emotion mapping ---

class TestEmotionMapping:
    def test_frustrated(self):
        assert _map_emotion("Frustrated") == ("Frustrated", 0.85)

    def test_angry(self):
        assert _map_emotion("Angry") == ("Frustrated", 0.85)

    def test_confused(self):
        assert _map_emotion("Confused") == ("Confused", 0.75)

    def test_uncertain(self):
        assert _map_emotion("Uncertain") == ("Confused", 0.75)

    def test_hesitant(self):
        assert _map_emotion("Hesitant") == ("Hesitant", 0.65)

    def test_anxious(self):
        assert _map_emotion("Anxious") == ("Hesitant", 0.65)

    def test_neutral(self):
        assert _map_emotion("Neutral") == ("Neutral", 0.2)

    def test_happy(self):
        assert _map_emotion("Happy") == ("Neutral", 0.2)

    def test_unknown_emotion(self):
        assert _map_emotion("SomethingElse") == ("Neutral", 0.3)


# --- Integration tests with mocked HTTP ---

SAMPLE_VELMA_RESPONSE = {
    "text": "I can't find the checkout button anywhere. This is so frustrating.",
    "duration_ms": 5000,
    "utterances": [
        {
            "utterance_uuid": "uuid-1",
            "text": "I can't find the checkout button anywhere.",
            "start_ms": 0,
            "duration_ms": 2500,
            "speaker": 1,
            "language": "en",
            "emotion": "Confused",
            "accent": "American",
        },
        {
            "utterance_uuid": "uuid-2",
            "text": "This is so frustrating.",
            "start_ms": 2500,
            "duration_ms": 2500,
            "speaker": 1,
            "language": "en",
            "emotion": "Frustrated",
            "accent": "American",
        },
    ],
}


def _mock_response(data: dict, status_code: int = 200):
    """Create a mock httpx response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = data
    mock.raise_for_status = MagicMock()
    return mock


@pytest.mark.asyncio
async def test_analyze_full_audio_success(tmp_path):
    """Velma 2 response is parsed — returns one result per utterance."""
    wav_file = tmp_path / "audio.wav"
    wav_file.write_bytes(b"RIFF" + b"\x00" * 100)

    mock_response = _mock_response(SAMPLE_VELMA_RESPONSE)

    with patch("modulate.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        results = await analyze_full_audio(str(wav_file))

    assert len(results) == 2
    # First utterance: Confused + text override ("can't find" → 0.75, but voice is also 0.75)
    assert results[0].sentiment == "Confused"
    assert results[0].score == 0.75
    assert results[0].timestamp == 0.0
    # Second utterance: Frustrated (voice = 0.85, text "frustrating" = 0.85)
    assert results[1].sentiment == "Frustrated"
    assert results[1].score == 0.85
    assert results[1].quote == "This is so frustrating."
    assert results[1].timestamp == 2.5


@pytest.mark.asyncio
async def test_analyze_full_audio_no_utterances(tmp_path):
    """Empty utterances list returns single result."""
    wav_file = tmp_path / "audio.wav"
    wav_file.write_bytes(b"RIFF" + b"\x00" * 100)

    empty_response = {"text": "", "duration_ms": 5000, "utterances": []}
    mock_response = _mock_response(empty_response)

    with patch("modulate.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        results = await analyze_full_audio(str(wav_file))

    assert len(results) == 1
    assert results[0].sentiment == "Neutral"
    assert results[0].score == 0.0


@pytest.mark.asyncio
async def test_analyze_full_audio_no_api_key(tmp_path):
    """Missing API key returns neutral placeholder without calling API."""
    wav_file = tmp_path / "audio.wav"
    wav_file.write_bytes(b"RIFF" + b"\x00" * 100)

    with patch.dict(os.environ, {"MODULATE_API_KEY": ""}):
        results = await analyze_full_audio(str(wav_file))

    assert len(results) == 1
    assert results[0].sentiment == "Neutral"
    assert results[0].score == 0.0


@pytest.mark.asyncio
async def test_analyze_full_audio_all_neutral(tmp_path):
    """All neutral utterances produce low friction scores."""
    wav_file = tmp_path / "audio.wav"
    wav_file.write_bytes(b"RIFF" + b"\x00" * 100)

    neutral_response = {
        "text": "Looks good to me.",
        "duration_ms": 3000,
        "utterances": [
            {
                "utterance_uuid": "uuid-1",
                "text": "Looks good to me.",
                "start_ms": 0,
                "duration_ms": 3000,
                "speaker": 1,
                "language": "en",
                "emotion": "Happy",
                "accent": "American",
            }
        ],
    }
    mock_response = _mock_response(neutral_response)

    with patch("modulate.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        results = await analyze_full_audio(str(wav_file))

    assert len(results) == 1
    assert results[0].sentiment == "Neutral"
    assert results[0].score == 0.2
    assert results[0].quote == "Looks good to me."
