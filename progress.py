"""
Progress — SSE event bus for real-time pipeline updates.

In-memory pub/sub: publish() pushes to all connected clients,
subscribe() yields SSE-formatted strings for StreamingResponse.
"""

import asyncio
import json
import time
from typing import AsyncGenerator

_subscribers: list[asyncio.Queue] = []


def publish(stage: str, message: str, detail: str | None = None):
    """Push a progress event to all connected SSE clients."""
    payload = {
        "stage": stage,
        "message": message,
        "time": time.time(),
    }
    if detail:
        payload["detail"] = detail

    for queue in _subscribers:
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass  # drop if client is slow


async def subscribe() -> AsyncGenerator[str, None]:
    """Async generator yielding SSE-formatted events. Auto-cleans up on disconnect."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    _subscribers.append(queue)
    try:
        while True:
            payload = await queue.get()
            yield f"data: {json.dumps(payload)}\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        _subscribers.remove(queue)
