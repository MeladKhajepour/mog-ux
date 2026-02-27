"""
Reka Client — Visual analysis of screenshots using the official Reka SDK.

Sends a screenshot frame to Reka Flash and asks it to identify
the UI element the user is struggling with.
"""

import base64
import json
import os

from reka.client import AsyncReka
from reka.types import ChatMessage, TypedText, TypedMediaContent

from models import VisualAnalysis

VISUAL_PROMPT = """\
You are a UX analyst. This screenshot was captured at the exact moment a user
expressed frustration or confusion during a usability test.

Context: {context}

Identify:
1. The specific UI element the user is most likely struggling with
2. The page or screen name

Respond in this exact JSON format:
{{"detected_element": "<element name>", "page": "<page/screen name>", "description": "<brief explanation of what's wrong>"}}
"""


async def analyze_screenshot(image_path: str, context: str = "") -> VisualAnalysis:
    """
    Send a screenshot to Reka Flash for visual UX analysis.

    Args:
        image_path: Path to the screenshot image file.
        context: Optional context about what the user was saying/feeling.

    Returns:
        VisualAnalysis with detected_element, page, and description.
    """
    api_key = os.getenv("REKA_API_KEY")
    if not api_key:
        print("[Reka] No REKA_API_KEY set — returning placeholder analysis")
        return VisualAnalysis(
            detected_element="Unknown Element",
            page="Unknown Page",
            description="Reka API key not configured; visual analysis skipped.",
        )

    # Read and base64-encode the image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Determine mime type from extension
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")

    prompt = VISUAL_PROMPT.format(context=context or "No additional context.")
    image_url = f"data:{mime_type};base64,{image_data}"

    client = AsyncReka(api_key=api_key)
    response = await client.chat.create(
        model="reka-flash",
        messages=[
            ChatMessage(
                role="user",
                content=[
                    TypedMediaContent(type="image_url", image_url=image_url),
                    TypedText(type="text", text=prompt),
                ],
            )
        ],
    )

    text = response.responses[0].message.content

    # Strip markdown fences if present
    if text.strip().startswith("```"):
        text = text.strip().split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    try:
        result = json.loads(text.strip())
    except json.JSONDecodeError:
        print(f"[Reka] Failed to parse response: {text}")
        return VisualAnalysis(
            detected_element="Unknown Element",
            page="Unknown Page",
            description=text.strip()[:200],
        )

    return VisualAnalysis(
        detected_element=result.get("detected_element", "Unknown Element"),
        page=result.get("page", "Unknown Page"),
        description=result.get("description", ""),
    )
