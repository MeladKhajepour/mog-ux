import asyncio
import json
import os

from google import genai

from models import FrictionEvent, Insight
from learner import recall_for_event

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
MODEL = "gemini-3-flash-preview"


DIAGNOSE_PROMPT = """You are a UX diagnostician. Analyze this friction event and diagnose the problem.

**Friction Event:**
- Timestamp: {timestamp}
- User sentiment: {sentiment} (score: {score})
- Visual context: "{detected_element}" on the "{page}" page
- User quote: "{user_quote}"

{past_learnings}

Diagnose the UX flaw. Do NOT suggest a fix — only identify the problem.
If this is a recurring issue based on past learnings, escalate severity accordingly.

Respond in this exact JSON format (no markdown, no code fences):
{{
  "root_cause": "specific diagnosis here",
  "severity": "critical|moderate|minor",
  "category": "navigation|visual_hierarchy|labeling|affordance|feedback|layout|accessibility|information_architecture"
}}"""


SUGGEST_FIX_PROMPT = """You are a UX designer generating a fix for a diagnosed issue.

**Diagnosed Problem:**
- Root cause: {root_cause}
- Severity: {severity}
- Category: {category}

**Industry Research (Yutori):**
{yutori_section}

Using the research above as your primary reference, suggest a specific and actionable fix for a design team.

Respond in this exact JSON format (no markdown, no code fences):
{{
  "suggested_fix": "actionable suggestion here"
}}"""


async def diagnose(event: FrictionEvent) -> Insight:
    """Phase 1: Diagnose root cause only — no fix suggestion yet."""
    try:
        past_learnings = await recall_for_event(event)
    except Exception as e:
        print(f"[Reflector] Memory recall failed (non-fatal): {e}")
        past_learnings = ""

    prompt = DIAGNOSE_PROMPT.format(
        timestamp=event.timestamp,
        sentiment=event.acoustic_data.sentiment,
        score=event.acoustic_data.score,
        detected_element=event.visual_context.detected_element,
        page=event.visual_context.page,
        user_quote=event.user_quote,
        past_learnings=past_learnings if past_learnings else "(No past learnings available yet — this is a fresh analysis.)",
    )

    print(f"[Reflector] Phase 1: Diagnosing with {MODEL}...")
    response = await asyncio.to_thread(
        _client.models.generate_content, model=MODEL, contents=prompt
    )
    text = response.text.strip()
    print(f"[Reflector] Diagnosis: {text[:100]}...")

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]

    parsed = json.loads(text)

    return Insight(
        event_id=event.event_id,
        friction_event=event,
        root_cause=parsed["root_cause"],
        severity=parsed["severity"],
        category=parsed["category"],
        suggested_fix="",
    )


async def suggest_fix(partial_insight: Insight, benchmarks: dict) -> Insight:
    """Phase 2: Generate a fix informed by Yutori benchmarks."""
    if benchmarks and benchmarks.get("recommendation"):
        yutori_section = (
            f"- Source: {benchmarks.get('source', 'Industry Research')}\n"
            f"- Best practice: {benchmarks['recommendation']}\n"
            f"- Real-world examples: {benchmarks.get('examples', 'N/A')}"
        )
    else:
        yutori_section = "(No benchmark data available — use your own UX knowledge.)"

    prompt = SUGGEST_FIX_PROMPT.format(
        root_cause=partial_insight.root_cause,
        severity=partial_insight.severity,
        category=partial_insight.category,
        yutori_section=yutori_section,
    )

    print(f"[Reflector] Phase 2: Generating Yutori-informed fix with {MODEL}...")
    response = await asyncio.to_thread(
        _client.models.generate_content, model=MODEL, contents=prompt
    )
    text = response.text.strip()
    print(f"[Reflector] Fix: {text[:100]}...")

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]

    parsed = json.loads(text)

    return Insight(
        event_id=partial_insight.event_id,
        friction_event=partial_insight.friction_event,
        root_cause=partial_insight.root_cause,
        severity=partial_insight.severity,
        category=partial_insight.category,
        suggested_fix=parsed["suggested_fix"],
    )
