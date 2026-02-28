"""
Yutori Client — UX benchmark research via the official Yutori Python SDK.

Creates research tasks through the Yutori Research API and polls
for structured results containing UX best practices.
"""

import asyncio
import os

from yutori import AsyncYutoriClient

YUTORI_API_KEY = os.getenv("YUTORI_API_KEY", "")

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string", "description": "The research source or authority"},
        "recommendation": {"type": "string", "description": "Actionable UX recommendation"},
        "examples": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Real-world examples of apps implementing this well",
        },
    },
    "required": ["source", "recommendation", "examples"],
}


async def search_benchmarks(issue_description: str, category: str) -> dict:
    """Query Yutori for UX best practices related to a diagnosed issue.

    Args:
        issue_description: The root cause diagnosis from the Reflector.
        category: The UX category (e.g. "navigation", "visual_hierarchy").

    Returns:
        Dict with keys: "source", "recommendation", "examples"
        Returns empty dict if the API call fails or no key is configured.
    """
    if not YUTORI_API_KEY:
        return {}

    query = (
        f"Research UX best practices for solving: {issue_description}. "
        f"Category: {category}. "
        f"Reference how top-tier apps handle this with specific examples."
    )

    try:
        client = AsyncYutoriClient(api_key=YUTORI_API_KEY)

        # Create a research task — returns a dict with task_id
        task = await client.research.create(
            query=query,
            output_schema=_OUTPUT_SCHEMA,
        )
        task_id = task["task_id"]

        # Poll for completion (max 30 seconds)
        max_polls = 15
        polls = 0
        while task.get("status") not in ("completed", "failed"):
            polls += 1
            if polls > max_polls:
                print(f"[Yutori] Timed out waiting for task {task_id}")
                await client.close()
                return {}
            await asyncio.sleep(2)
            task = await client.research.get(task_id)

        await client.close()

        if task.get("status") == "failed":
            print(f"[Yutori] Research task failed: {task_id}")
            return {}

        # Extract structured result from the completed task
        result = task.get("output", {})
        return {
            "source": result.get("source", "Yutori Research"),
            "recommendation": result.get("recommendation", ""),
            "examples": result.get("examples", []),
        }
    except Exception as e:
        print(f"[Yutori] API call failed: {e}")
        return {}
