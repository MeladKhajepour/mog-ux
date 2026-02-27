"""Tests for yutori_client.py — Yutori Research API integration."""

import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

os.environ.setdefault("YUTORI_API_KEY", "test-key")

from yutori_client import search_benchmarks


@pytest.mark.asyncio
async def test_search_benchmarks_success():
    """Yutori returns completed research task — parsed into dict."""
    create_response = {
        "task_id": "task-123",
        "status": "processing",
    }
    completed_response = {
        "task_id": "task-123",
        "status": "completed",
        "output": {
            "source": "Nielsen Norman Group",
            "recommendation": "Use progressive disclosure for complex forms",
            "examples": ["Stripe Checkout", "Shopify Cart"],
        },
    }

    mock_client = AsyncMock()
    mock_client.research.create.return_value = create_response
    mock_client.research.get.return_value = completed_response
    mock_client.close = AsyncMock()

    with patch("yutori_client.AsyncYutoriClient", return_value=mock_client):
        result = await search_benchmarks("confusing checkout flow", "navigation")

    assert result["source"] == "Nielsen Norman Group"
    assert result["recommendation"] == "Use progressive disclosure for complex forms"
    assert "Stripe Checkout" in result["examples"]

    mock_client.research.create.assert_called_once()
    mock_client.research.get.assert_called_once_with("task-123")
    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_search_benchmarks_immediate_completion():
    """Task completes immediately (status=completed in create response)."""
    create_response = {
        "task_id": "task-456",
        "status": "completed",
        "output": {
            "source": "Baymard Institute",
            "recommendation": "Add breadcrumbs",
            "examples": ["Amazon"],
        },
    }

    mock_client = AsyncMock()
    mock_client.research.create.return_value = create_response
    mock_client.close = AsyncMock()

    with patch("yutori_client.AsyncYutoriClient", return_value=mock_client):
        result = await search_benchmarks("poor navigation", "navigation")

    assert result["source"] == "Baymard Institute"
    # Should not poll since already completed
    mock_client.research.get.assert_not_called()


@pytest.mark.asyncio
async def test_search_benchmarks_task_failed():
    """Failed research task returns empty dict."""
    create_response = {"task_id": "task-789", "status": "processing"}
    failed_response = {"task_id": "task-789", "status": "failed"}

    mock_client = AsyncMock()
    mock_client.research.create.return_value = create_response
    mock_client.research.get.return_value = failed_response
    mock_client.close = AsyncMock()

    with patch("yutori_client.AsyncYutoriClient", return_value=mock_client):
        result = await search_benchmarks("some issue", "labeling")

    assert result == {}


@pytest.mark.asyncio
async def test_search_benchmarks_no_api_key():
    """No API key returns empty dict without calling API."""
    with patch.dict(os.environ, {"YUTORI_API_KEY": ""}):
        # Need to reload the module to pick up the empty env var
        import importlib
        import yutori_client
        importlib.reload(yutori_client)

        result = await yutori_client.search_benchmarks("some issue", "navigation")

    assert result == {}

    # Restore the key for other tests
    os.environ["YUTORI_API_KEY"] = "test-key"
    importlib.reload(yutori_client)


@pytest.mark.asyncio
async def test_search_benchmarks_api_error():
    """API exception returns empty dict gracefully."""
    mock_client = AsyncMock()
    mock_client.research.create.side_effect = Exception("Connection timeout")

    with patch("yutori_client.AsyncYutoriClient", return_value=mock_client):
        result = await search_benchmarks("some issue", "visual_hierarchy")

    assert result == {}
