import json
import os

import google.generativeai as genai

from models import FrictionEvent, Insight
from learner import recall_for_event
from yutori_client import search_benchmarks

genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

model = genai.GenerativeModel("gemini-3-flash-preview")


REFLECTOR_PROMPT = """You are a UX diagnostician. Analyze this friction event from a user testing session and provide a structured diagnosis.

**Friction Event:**
- Timestamp: {timestamp}
- User sentiment: {sentiment} (confidence: {score})
- Visual context: User is looking at "{detected_element}" on the "{page}" page
- User quote: "{user_quote}"

{past_learnings}

**Your task:**
1. Consider any past learnings above — if this is a recurring issue, escalate severity and reference the pattern.
2. Diagnose the specific qualitative UI flaw causing this friction.
3. Classify the severity as "critical", "moderate", or "minor".
4. Assign a UX category from: "navigation", "visual_hierarchy", "labeling", "affordance", "feedback", "layout", "accessibility", "information_architecture".
5. Suggest a specific, actionable fix for a design team.

**Respond in this exact JSON format (no markdown, no code fences):**
{{
  "root_cause": "specific diagnosis here",
  "severity": "critical|moderate|minor",
  "category": "one of the categories above",
  "suggested_fix": "actionable suggestion here"
}}"""


async def reflect(event: FrictionEvent) -> Insight:
    """Send a friction event to Gemini for root cause analysis, enriched with past learnings."""
    # Recall relevant memories from previous sessions
    past_learnings = recall_for_event(event)

    prompt = REFLECTOR_PROMPT.format(
        timestamp=event.timestamp,
        sentiment=event.acoustic_data.sentiment,
        score=event.acoustic_data.score,
        detected_element=event.visual_context.detected_element,
        page=event.visual_context.page,
        user_quote=event.user_quote,
        past_learnings=past_learnings if past_learnings else "(No past learnings available yet — this is a fresh analysis.)",
    )

    response = await model.generate_content_async(prompt)
    text = response.text.strip()

    # Strip markdown code fences if Gemini adds them
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
        suggested_fix=parsed["suggested_fix"],
    )


async def reflect_with_benchmarks(event: FrictionEvent) -> tuple[Insight, dict]:
    """Reflect on a friction event and fetch Yutori benchmarks."""
    insight = await reflect(event)
    benchmarks = await search_benchmarks(insight.root_cause, insight.category)
    return insight, benchmarks
