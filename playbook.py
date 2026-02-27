import json
import os
from datetime import datetime, timezone

from models import Bullet, Playbook

PLAYBOOK_PATH = os.path.join(os.path.dirname(__file__), "playbook.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_playbook() -> Playbook:
    """Load the playbook from disk, or create an empty one."""
    if os.path.exists(PLAYBOOK_PATH):
        with open(PLAYBOOK_PATH, "r") as f:
            return Playbook(**json.load(f))
    return Playbook(session_id="default", bullets=[], last_updated=_now())


def save_playbook(playbook: Playbook) -> None:
    """Atomically write playbook to disk."""
    playbook.last_updated = _now()
    tmp_path = PLAYBOOK_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(playbook.model_dump(), f, indent=2)
    os.replace(tmp_path, PLAYBOOK_PATH)


def _keyword_set(text: str) -> set[str]:
    """Extract lowercase keywords from text, ignoring short words."""
    return {w.lower() for w in text.split() if len(w) > 3}


def _keyword_overlap(a: str, b: str) -> float:
    """Return 0-1 overlap ratio between two strings' keyword sets."""
    set_a = _keyword_set(a)
    set_b = _keyword_set(b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    smaller = min(len(set_a), len(set_b))
    return len(intersection) / smaller if smaller > 0 else 0.0


def find_matching_bullet(playbook: Playbook, category: str, title: str) -> Bullet | None:
    """Find an existing bullet with the same category and similar title."""
    for bullet in playbook.bullets:
        if bullet.category == category and _keyword_overlap(bullet.title, title) > 0.6:
            return bullet
    return None


def add_or_merge_bullet(playbook: Playbook, new_bullet: Bullet) -> Playbook:
    """Delta-merge a bullet into the playbook. Returns the updated playbook."""
    existing = find_matching_bullet(playbook, new_bullet.category, new_bullet.title)

    if existing:
        existing.friction_count += new_bullet.friction_count
        existing.evidence.extend(new_bullet.evidence)
        # Keep the higher severity
        severity_rank = {"critical": 3, "moderate": 2, "minor": 1}
        if severity_rank.get(new_bullet.severity, 0) > severity_rank.get(existing.severity, 0):
            existing.severity = new_bullet.severity
        existing.updated_at = _now()
        # Merge content if new bullet has additional info
        if new_bullet.content not in existing.content:
            existing.content += f" | {new_bullet.content}"
        # Preserve image URLs (use latest if existing are empty)
        if new_bullet.frame_url and not existing.frame_url:
            existing.frame_url = new_bullet.frame_url
        if new_bullet.mockup_url and not existing.mockup_url:
            existing.mockup_url = new_bullet.mockup_url
    else:
        playbook.bullets.append(new_bullet)

    save_playbook(playbook)
    return playbook
