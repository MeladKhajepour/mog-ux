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


def curate_friction_log(insight: Insight, frame_url: str = "") -> None:
    """Phase 1: Emit only the friction_log bullet. Called immediately after diagnose()."""
    playbook = load_playbook()
    evidence = _make_evidence(insight)
    now = _now()

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
        mockup_url="",
        created_at=now,
        updated_at=now,
    )
    add_or_merge_bullet(playbook, friction_bullet)


def curate_strategy(insight: Insight, benchmarks: dict, frame_url: str = "") -> None:
    """Phase 2: Emit hard_strategy + benchmark bullets. Called after suggest_fix()."""
    playbook = load_playbook()
    evidence = _make_evidence(insight)
    now = _now()

    # Hard Strategy bullet
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
        mockup_url="",
        created_at=now,
        updated_at=now,
    )
    playbook = add_or_merge_bullet(playbook, strategy_bullet)

    # Benchmark bullet (only if Yutori returned data)
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


def curate(insight: Insight, benchmarks: dict, frame_url: str = "", mockup_url: str = "") -> None:
    """Convenience wrapper — calls both phases. Keeps existing callers working."""
    curate_friction_log(insight, frame_url)
    curate_strategy(insight, benchmarks, frame_url)
