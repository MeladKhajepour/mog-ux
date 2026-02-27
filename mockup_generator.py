"""
Mockup Generator — Nano Banana Pro image editing.

Takes an extracted frame + diagnosed problem + suggested fix,
sends to Gemini to generate a mockup showing the improvement.
"""

import os

from google import genai
from google.genai import types
from PIL import Image


async def generate_mockup(frame_path: str, problem: str, suggestion: str) -> str:
    """
    Generate a UI mockup showing the suggested fix applied to the original frame.

    Returns path to the saved mockup image.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    image = Image.open(frame_path)

    prompt = f"""You are a UI/UX designer. This screenshot shows a real app interface.

Problem identified: {problem}
Suggested fix: {suggestion}

Generate a modified version of this screenshot that applies the suggested fix.
Keep the overall layout and design language identical — only modify the specific
element mentioned. Make the change look natural and production-ready."""

    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[prompt, image],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )

    mockup_path = frame_path.replace(".jpg", "_mockup.png")
    for part in response.parts:
        if part.inline_data is not None:
            part.as_image().save(mockup_path)
            break

    return mockup_path
