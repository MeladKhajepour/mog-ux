import uuid
from datetime import datetime, timezone

from models import Bullet, Insight
from playbook import add_or_merge_bullet, load_playbook


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_evidence(insight: Insight) -> str:
    """Build a human-readable evidence string from an insight."""
    e = insight.friction_event
    return (
        f"[{e.timestamp}] {e.acoustic_data.sentiment} "
        f"(score: {e.acoustic_data.score}) on {e.visual_context.page} page — "
        f'"{e.user_quote}"'
    )


def curate(insight: Insight, benchmarks: dict, frame_url: str = "", mockup_url: str = "") -> None:
    """Delta-merge an insight (and optional benchmarks) into the playbook.

    Creates up to 3 bullets per insight:
    1. friction_log — what went wrong
    2. hard_strategy — how to fix it
    3. benchmark — industry best practice (if Yutori returned data)
    """
    playbook = load_playbook()
    evidence = _make_evidence(insight)
    now = _now()

    # 1. Friction Log bullet
    friction_bullet = Bullet(
        id=str(uuid.uuid4()),
        bullet_type="friction_log",
        category=insight.category,
        title=f"{insight.category}: {insight.root_cause}",
        content=insight.root_cause,
        evidence=[evidence],
        friction_count=1,
        severity=insight.severity,
        benchmark_source="",
        frame_url=frame_url,
        mockup_url=mockup_url,
        created_at=now,
        updated_at=now,
    )
    playbook = add_or_merge_bullet(playbook, friction_bullet)

    # 2. Hard Strategy bullet
    strategy_bullet = Bullet(
        id=str(uuid.uuid4()),
        bullet_type="hard_strategy",
        category=insight.category,
        title=f"Fix: {insight.suggested_fix}",
        content=insight.suggested_fix,
        evidence=[evidence],
        friction_count=1,
        severity=insight.severity,
        benchmark_source="",
        frame_url=frame_url,
        mockup_url=mockup_url,
        created_at=now,
        updated_at=now,
    )
    playbook = add_or_merge_bullet(playbook, strategy_bullet)

    # 3. Benchmark bullet (only if Yutori returned data)
    if benchmarks and benchmarks.get("recommendation"):
        benchmark_bullet = Bullet(
            id=str(uuid.uuid4()),
            bullet_type="benchmark",
            category=insight.category,
            title=f"Benchmark: {insight.category} — {benchmarks.get('source', 'Industry')}",
            content=benchmarks["recommendation"],
            evidence=[evidence],
            friction_count=1,
            severity=insight.severity,
            benchmark_source=benchmarks.get("source", "Yutori Research"),
            created_at=now,
            updated_at=now,
        )
        add_or_merge_bullet(playbook, benchmark_bullet)
