"""Tests for reka_client.py — Reka SDK integration."""

import json
import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

os.environ.setdefault("REKA_API_KEY", "test-key")

from reka_client import analyze_screenshot
from models import VisualAnalysis


def _make_reka_response(content_text: str):
    """Build a mock Reka ChatResponse."""
    message = MagicMock()
    message.content = content_text

    msg_response = MagicMock()
    msg_response.message = message

    response = MagicMock()
    response.responses = [msg_response]
    return response


@pytest.mark.asyncio
async def test_analyze_screenshot_success(tmp_path):
    """Reka returns valid JSON — parsed into VisualAnalysis."""
    img = tmp_path / "frame.png"
    img.write_bytes(b"\x89PNG" + b"\x00" * 100)

    reka_json = json.dumps({
        "detected_element": "Checkout Button",
        "page": "Cart Page",
        "description": "Button is hidden below the fold",
    })
    mock_response = _make_reka_response(reka_json)

    with patch("reka_client.AsyncReka") as mock_reka_cls:
        mock_client = AsyncMock()
        mock_client.chat.create.return_value = mock_response
        mock_reka_cls.return_value = mock_client

        result = await analyze_screenshot(str(img), context="User seemed frustrated")

    assert isinstance(result, VisualAnalysis)
    assert result.detected_element == "Checkout Button"
    assert result.page == "Cart Page"
    assert result.description == "Button is hidden below the fold"

    # Verify SDK was called with correct model and message structure
    mock_client.chat.create.assert_called_once()
    call_kwargs = mock_client.chat.create.call_args.kwargs
    assert call_kwargs["model"] == "reka-flash"
    assert len(call_kwargs["messages"]) == 1
    assert call_kwargs["messages"][0].role == "user"


@pytest.mark.asyncio
async def test_analyze_screenshot_markdown_fences(tmp_path):
    """Reka wraps JSON in markdown fences — still parsed correctly."""
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

    reka_text = '```json\n{"detected_element": "Nav Bar", "page": "Home", "description": "Nav is confusing"}\n```'
    mock_response = _make_reka_response(reka_text)

    with patch("reka_client.AsyncReka") as mock_reka_cls:
        mock_client = AsyncMock()
        mock_client.chat.create.return_value = mock_response
        mock_reka_cls.return_value = mock_client

        result = await analyze_screenshot(str(img))

    assert result.detected_element == "Nav Bar"
    assert result.page == "Home"


@pytest.mark.asyncio
async def test_analyze_screenshot_invalid_json(tmp_path):
    """Reka returns non-JSON text — falls back gracefully."""
    img = tmp_path / "frame.png"
    img.write_bytes(b"\x89PNG" + b"\x00" * 100)

    mock_response = _make_reka_response("I see a confusing interface with many buttons")

    with patch("reka_client.AsyncReka") as mock_reka_cls:
        mock_client = AsyncMock()
        mock_client.chat.create.return_value = mock_response
        mock_reka_cls.return_value = mock_client

        result = await analyze_screenshot(str(img))

    assert result.detected_element == "Unknown Element"
    assert result.page == "Unknown Page"
    assert "confusing interface" in result.description


@pytest.mark.asyncio
async def test_analyze_screenshot_no_api_key(tmp_path):
    """Missing API key returns placeholder without calling Reka."""
    img = tmp_path / "frame.png"
    img.write_bytes(b"\x89PNG" + b"\x00" * 100)

    with patch.dict(os.environ, {"REKA_API_KEY": ""}):
        result = await analyze_screenshot(str(img))

    assert result.detected_element == "Unknown Element"
    assert "not configured" in result.description
