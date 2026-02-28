"""
Learner — Self-learning agent powered by mem0.

Three learning loops:
1. Per-event: stores each curated insight as a memory
2. Session summary: extracts cross-event patterns after all chunks are processed
3. Recall: retrieves relevant past learnings before the Reflector analyzes a new event

Uses Gemini for both LLM and embeddings (no OpenAI dependency).
"""

import os

from mem0 import Memory

from models import FrictionEvent, Insight

_memory: Memory | None = None

# Stable user_id so all memories accumulate for this system
USER_ID = "mog-ux-agent"


def _get_memory() -> Memory:
    """Lazy-init mem0 client configured with Gemini for LLM + embeddings."""
    global _memory
    if _memory is None:
        config = {
            "version": "v1.1",
            "llm": {
                "provider": "gemini",
                "config": {
                    "model": "gemini-3-flash-preview",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                },
            },
            "embedder": {
                "provider": "gemini",
                "config": {
                    "model": "models/gemini-embedding-001",
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mog_ux_memories",
                    "embedding_model_dims": 768,
                    "on_disk": True,
                    "path": os.path.join(os.path.dirname(__file__), ".mem0_store"),
                },
            },
        }
        _memory = Memory.from_config(config)
    return _memory


def store_insight(insight: Insight) -> None:
    """Store a curated insight as a mem0 memory (per-event learning)."""
    mem = _get_memory()

    message = (
        f"{insight.severity.upper()} {insight.category} issue on "
        f"{insight.friction_event.visual_context.page} page — "
        f"element: {insight.friction_event.visual_context.detected_element}. "
        f"Root cause: {insight.root_cause}. "
        f"Suggested fix: {insight.suggested_fix}. "
        f'User quote: "{insight.friction_event.user_quote}"'
    )

    metadata = {
        "type": "insight",
        "category": insight.category,
        "severity": insight.severity,
        "page": insight.friction_event.visual_context.page,
        "element": insight.friction_event.visual_context.detected_element,
    }

    mem.add(message, user_id=USER_ID, metadata=metadata)
    print(f"[Learner] Stored insight: {insight.severity} {insight.category} on {insight.friction_event.visual_context.page}")


def store_session_summary(events: list[FrictionEvent]) -> None:
    """After all chunks from one upload are processed, store a session-level summary."""
    if not events:
        return

    mem = _get_memory()

    # Build a summary of the session's friction events
    page_counts: dict[str, int] = {}
    sentiments: list[str] = []

    for event in events:
        page = event.visual_context.page
        sentiment = event.acoustic_data.sentiment
        page_counts[page] = page_counts.get(page, 0) + 1
        sentiments.append(sentiment)

    # Identify dominant patterns
    top_page = max(page_counts, key=page_counts.get) if page_counts else "unknown"
    total = len(events)

    summary_parts = [
        f"Session processed {total} friction events.",
        f"Most problematic page: {top_page} ({page_counts.get(top_page, 0)}/{total} events).",
    ]

    # Page breakdown
    for page, count in sorted(page_counts.items(), key=lambda x: -x[1]):
        summary_parts.append(f"  - {page}: {count} friction events")

    # Sentiment breakdown
    sentiment_counts: dict[str, int] = {}
    for s in sentiments:
        sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
    dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else "Neutral"
    summary_parts.append(f"Dominant user sentiment: {dominant_sentiment}.")

    message = " ".join(summary_parts)

    mem.add(
        message,
        user_id=USER_ID,
        metadata={"type": "session_summary", "event_count": total},
    )
    print(f"[Learner] Stored session summary: {total} events, top page: {top_page}")


def get_all_memories() -> list[dict]:
    """Return all stored memories for the memories viewer."""
    mem = _get_memory()
    results = mem.get_all(user_id=USER_ID)
    # mem0 returns {"results": [...]} or a list depending on version
    if isinstance(results, dict):
        return results.get("results", results.get("memories", []))
    return results


def recall_for_event(event: FrictionEvent) -> str:
    """Retrieve relevant past learnings for a new friction event."""
    mem = _get_memory()

    query = (
        f"{event.acoustic_data.sentiment} issue on {event.visual_context.page} page, "
        f"element: {event.visual_context.detected_element}. "
        f'User said: "{event.user_quote}"'
    )

    results = mem.search(query, user_id=USER_ID, limit=5)

    if not results.get("results"):
        return ""

    memories = results["results"]
    lines = ["PAST LEARNINGS (from previous sessions):"]
    for i, m in enumerate(memories, 1):
        lines.append(f"{i}. {m['memory']}")

    context = "\n".join(lines)
    print(f"[Learner] Recalled {len(memories)} memories for event on {event.visual_context.page}")
    return context
