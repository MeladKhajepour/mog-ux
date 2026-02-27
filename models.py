from pydantic import BaseModel
from typing import Optional


# --- Sensing Pipeline Models ---

class AudioSegment(BaseModel):
    chunk_index: int
    start_time: float   # seconds into the video
    end_time: float
    file_path: str      # path to the .wav chunk


class SentimentResult(BaseModel):
    sentiment: str      # "Frustrated", "Confused", "Hesitant", "Neutral"
    score: float        # 0.0 - 1.0
    quote: str          # Direct quote from the user
    timestamp: float    # Seconds into the original video
    chunk_index: int
    voice_features: dict = {}  # Detailed vocal breakdown: pitch, pace, volume, etc.


class VisualAnalysis(BaseModel):
    detected_element: str   # e.g. "Primary Action Button"
    page: str               # e.g. "Checkout"
    description: str        # Full description from Reka


# --- Shared Contract (incoming from partner's Sensing module) ---

class AcousticData(BaseModel):
    sentiment: str  # "Frustrated", "Confused", "Hesitant"
    score: float    # 0.0 - 1.0


class VisualContext(BaseModel):
    detected_element: str  # e.g. "Primary Action Button"
    page: str              # e.g. "Checkout"


class FrictionEvent(BaseModel):
    event_id: str
    timestamp: str
    acoustic_data: AcousticData
    visual_context: VisualContext
    user_quote: str
    status: str  # "pending_reflection"
    frame_path: str = ""  # path to extracted frame at friction spike


# --- Internal Models (produced by Brain module) ---

class Insight(BaseModel):
    event_id: str
    friction_event: FrictionEvent
    root_cause: str      # Gemini's specific diagnosis
    severity: str        # "critical" | "moderate" | "minor"
    category: str        # "navigation", "visual_hierarchy", "labeling", etc.
    suggested_fix: str   # Qualitative suggestion


class Bullet(BaseModel):
    id: str                    # UUID
    bullet_type: str           # "hard_strategy" | "friction_log" | "benchmark"
    category: str              # UX category: "navigation", "visual_hierarchy", etc.
    title: str
    content: str
    evidence: list[str]        # Timestamps + user quotes
    friction_count: int        # How many times this issue appeared
    severity: str
    benchmark_source: str      # Yutori source, empty string if N/A
    frame_url: str = ""        # URL to original frame (served via /uploads)
    mockup_url: str = ""       # URL to Nano Banana generated mockup
    created_at: str
    updated_at: str


class Playbook(BaseModel):
    session_id: str
    bullets: list[Bullet]
    last_updated: str
