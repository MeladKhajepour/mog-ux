"""
Modulate — Voice-first prosodic analysis.

Uses the Modulate Velma 2 STT Batch API to analyze audio for real
emotion signals from the waveform, with speaker diarization.
"""

import os

import httpx

from models import SentimentResult

VELMA_URL = "https://modulate-developer-apis.com/api/velma-2-stt-batch"

# Modulate emotion → (our sentiment label, friction score)
_EMOTION_MAP = {
    "Frustrated": ("Frustrated", 0.85),
    "Angry": ("Frustrated", 0.85),
    "Confused": ("Confused", 0.75),
    "Uncertain": ("Confused", 0.75),
    "Hesitant": ("Hesitant", 0.65),
    "Anxious": ("Hesitant", 0.65),
    "Neutral": ("Neutral", 0.2),
    "Happy": ("Neutral", 0.2),
    "Calm": ("Neutral", 0.2),
}


def _map_emotion(emotion: str) -> tuple[str, float]:
    """Map a Modulate emotion label to our sentiment category and score."""
    return _EMOTION_MAP.get(emotion, ("Neutral", 0.3))


async def analyze_sentiment(audio_path: str, chunk_index: int, start_time: float) -> SentimentResult:
    """
    Analyze a single audio chunk for user sentiment via Modulate Velma 2.

    Args:
        audio_path: Path to the .wav audio chunk.
        chunk_index: Index of this chunk in the sequence.
        start_time: When this chunk starts in the original video (seconds).

    Returns:
        SentimentResult with sentiment, score, quote, and timestamp.
    """
    api_key = os.getenv("MODULATE_API_KEY")
    if not api_key:
        print("[Modulate] No MODULATE_API_KEY set — returning neutral placeholder")
        return SentimentResult(
            sentiment="Neutral",
            score=0.0,
            quote="",
            timestamp=start_time,
            chunk_index=chunk_index,
            voice_features={},
        )

    # Send audio to Velma 2 STT Batch API
    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(audio_path, "rb") as f:
            response = await client.post(
                VELMA_URL,
                headers={"X-API-Key": api_key},
                files={"upload_file": (os.path.basename(audio_path), f, "audio/wav")},
                data={
                    "speaker_diarization": "true",
                    "emotion_signal": "true",
                },
            )
        response.raise_for_status()
        data = response.json()

    utterances = data.get("utterances", [])
    if not utterances:
        print(f"[Modulate] Chunk {chunk_index}: no utterances detected")
        return SentimentResult(
            sentiment="Neutral",
            score=0.0,
            quote=data.get("text", ""),
            timestamp=start_time,
            chunk_index=chunk_index,
            voice_features={},
        )

    # Map each utterance's emotion and find the highest-friction one
    best_utterance = None
    best_score = -1.0
    emotion_counts: dict[str, int] = {}

    for utt in utterances:
        raw_emotion = utt.get("emotion", "Neutral")
        sentiment, score = _map_emotion(raw_emotion)
        emotion_counts[raw_emotion] = emotion_counts.get(raw_emotion, 0) + 1

        if score > best_score:
            best_score = score
            best_utterance = utt
            best_sentiment = sentiment

    # Extract quote and timestamp from the most friction-relevant utterance
    quote = best_utterance.get("text", "") if best_utterance else ""
    utterance_start_ms = best_utterance.get("start_ms", 0) if best_utterance else 0
    timestamp = start_time + (utterance_start_ms / 1000.0)

    # Build voice_features from emotion distribution across utterances
    voice_features = {
        "emotion_counts": emotion_counts,
        "utterance_count": len(utterances),
        "dominant_emotion": best_utterance.get("emotion", "Neutral") if best_utterance else "Neutral",
        "full_transcription": data.get("text", ""),
        "duration_ms": data.get("duration_ms", 0),
    }

    print(f"[Modulate] Chunk {chunk_index}: {best_sentiment} ({best_score:.2f}) — emotions: {emotion_counts}")

    return SentimentResult(
        sentiment=best_sentiment,
        score=best_score,
        quote=quote,
        timestamp=timestamp,
        chunk_index=chunk_index,
        voice_features=voice_features,
    )
