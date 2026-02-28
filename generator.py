"""
Generator — Sensing pipeline.

Takes an uploaded video, extracts audio, analyzes the full audio
for per-utterance sentiment, extracts frames at friction spikes,
performs visual analysis, and emits FrictionEvents into the Brain pipeline queue.
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone

from models import FrictionEvent, AcousticData, VisualContext
from modulate import analyze_full_audio
from reka_client import analyze_screenshot
from learner import store_session_summary
from progress import publish

FRICTION_THRESHOLD = 0.6


async def _run_ffmpeg(args: list[str]):
    """Run an ffmpeg command asynchronously."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")
    return stdout, stderr


async def extract_audio(video_path: str, output_dir: str) -> str:
    """Extract audio track from video as WAV."""
    audio_path = os.path.join(output_dir, "audio.wav")
    await _run_ffmpeg([
        "-i", video_path,
        "-vn",                  # no video
        "-acodec", "pcm_s16le", # standard WAV
        "-ar", "16000",         # 16kHz sample rate
        "-ac", "1",             # mono
        "-y",                   # overwrite
        audio_path,
    ])
    print(f"[Generator] Audio extracted → {audio_path}")
    publish("audio_extract", "Audio track extracted")
    return audio_path


async def extract_frame(video_path: str, timestamp: float, output_dir: str) -> str:
    """Extract a single frame from the video at the given timestamp."""
    frame_path = os.path.join(output_dir, f"frame_{timestamp:.1f}.jpg")
    await _run_ffmpeg([
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-y",
        frame_path,
    ])
    print(f"[Generator] Frame extracted at {timestamp:.1f}s → {frame_path}")
    return frame_path


async def process_upload(video_path: str, event_queue: asyncio.Queue):
    """
    Full sensing pipeline: video → audio → sentiment → frames → FrictionEvents.

    Sends the entire audio to Modulate for per-utterance analysis,
    then processes each friction utterance individually.
    """
    work_dir = video_path + "_work"
    os.makedirs(work_dir, exist_ok=True)

    filename = os.path.basename(video_path)
    filesize_mb = os.path.getsize(video_path) / (1024 * 1024)
    publish("upload", f"Video received: {filename} ({filesize_mb:.1f}MB)")
    print(f"[Generator] Starting pipeline for {video_path}")

    # 1. Extract audio
    publish("audio_extract", "Extracting audio track...")
    audio_path = await extract_audio(video_path, work_dir)

    # 2. Analyze full audio — get per-utterance results
    publish("analyzing_audio", "Analyzing full audio for sentiment...")
    results = await analyze_full_audio(audio_path)
    publish("voice_analysis", f"Found {len(results)} utterances")
    print(f"[Generator] {len(results)} utterances analyzed")

    # 3. Process each friction utterance
    friction_count = 0
    session_events: list[FrictionEvent] = []

    for result in results:
        if result.score <= FRICTION_THRESHOLD:
            continue

        friction_count += 1
        publish("friction_spike", f"FRICTION at {result.timestamp:.1f}s — {result.sentiment} ({result.score:.2f})", result.quote)
        print(f"[Generator] FRICTION at {result.timestamp:.1f}s — {result.sentiment} ({result.score:.2f})")

        # Extract frame at the friction timestamp
        frame_path = await extract_frame(video_path, result.timestamp, work_dir)

        # Visual analysis of the frame
        context = f"User said: \"{result.quote}\" (sentiment: {result.sentiment}, score: {result.score:.2f})"
        visual = await analyze_screenshot(frame_path, context)
        publish("visual_analysis", f"Visual: {visual.detected_element} on {visual.page}")
        print(f"[Generator] Visual: {visual.detected_element} on {visual.page}")

        # Build and emit FrictionEvent
        event = FrictionEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            acoustic_data=AcousticData(
                sentiment=result.sentiment,
                score=result.score,
            ),
            visual_context=VisualContext(
                detected_element=visual.detected_element,
                page=visual.page,
            ),
            user_quote=result.quote,
            status="pending_reflection",
            frame_path=frame_path,
        )

        session_events.append(event)
        await event_queue.put(event)
        publish("event_queued", "FrictionEvent queued")
        print(f"[Generator] FrictionEvent {event.event_id} queued")

    # 4. Store session summary for cross-session learning
    if session_events:
        try:
            store_session_summary(session_events)
        except Exception as e:
            print(f"[Generator] Failed to store session summary: {e}")

    publish("session_done", f"Pipeline complete: {friction_count} friction events from {len(results)} utterances")
    print(f"[Generator] Pipeline complete: {friction_count} friction events from {len(results)} utterances")
