"""
Modulate — Voice-first prosodic analysis.

Uses the Modulate Velma 2 STT Batch API to analyze audio for real
emotion signals from the waveform, with speaker diarization.

Sends the full audio file and returns per-utterance friction results.
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


# Text-based friction signals — catches frustration/confusion the voice tone misses
_FRICTION_PHRASES = [
    ("can't figure", "Confused", 0.80),
    ("can't seem to", "Confused", 0.75),
    ("don't see", "Confused", 0.75),
    ("don't know how", "Confused", 0.75),
    ("confusing", "Confused", 0.80),
    ("confused", "Confused", 0.80),
    ("where is", "Confused", 0.70),
    ("where do i", "Confused", 0.70),
    ("how do i", "Confused", 0.65),
    ("not working", "Frustrated", 0.80),
    ("doesn't work", "Frustrated", 0.80),
    ("broken", "Frustrated", 0.85),
    ("frustrating", "Frustrated", 0.85),
    ("annoying", "Frustrated", 0.80),
    ("what the", "Frustrated", 0.75),
    ("makes no sense", "Frustrated", 0.80),
    ("no idea", "Confused", 0.75),
    ("impossible", "Frustrated", 0.85),
    ("stuck", "Frustrated", 0.75),
    ("give up", "Frustrated", 0.90),
]


def _text_friction_check(text: str) -> tuple[str, float]:
    """Scan transcript text for friction phrases. Returns (sentiment, score) or ("Neutral", 0.0)."""
    lower = text.lower()
    best_sentiment = "Neutral"
    best_score = 0.0
    for phrase, sentiment, score in _FRICTION_PHRASES:
        if phrase in lower and score > best_score:
            best_sentiment = sentiment
            best_score = score
    return best_sentiment, best_score


async def analyze_full_audio(audio_path: str) -> list[SentimentResult]:
    """
    Analyze the full audio file via Modulate Velma 2.

    Returns a list of SentimentResults — one per utterance that has friction,
    with accurate timestamps from the API.
    """
    api_key = os.getenv("MODULATE_API_KEY")
    if not api_key:
        print("[Modulate] No MODULATE_API_KEY set — returning neutral placeholder")
        return [SentimentResult(
            sentiment="Neutral",
            score=0.0,
            quote="",
            timestamp=0.0,
            chunk_index=0,
            voice_features={},
        )]

    # Send full audio to Velma 2 STT Batch API
    async with httpx.AsyncClient(timeout=300.0) as client:
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
    full_text = data.get("text", "")

    if not utterances:
        print("[Modulate] No utterances detected in audio")
        # Still check the full transcript text for friction
        text_sentiment, text_score = _text_friction_check(full_text)
        return [SentimentResult(
            sentiment=text_sentiment if text_score > 0 else "Neutral",
            score=text_score,
            quote=full_text,
            timestamp=0.0,
            chunk_index=0,
            voice_features={"full_transcription": full_text},
        )]

    print(f"[Modulate] {len(utterances)} utterances detected")

    # Process each utterance individually
    results: list[SentimentResult] = []
    for i, utt in enumerate(utterances):
        raw_emotion = utt.get("emotion", "Neutral")
        sentiment, score = _map_emotion(raw_emotion)
        utt_text = utt.get("text", "")
        start_ms = utt.get("start_ms", 0)
        timestamp = start_ms / 1000.0

        # Text-based friction check on this utterance's text
        text_sentiment, text_score = _text_friction_check(utt_text)
        if text_score > score:
            print(f"[Modulate] Utterance {i}: text override — voice={sentiment}({score:.2f}), text={text_sentiment}({text_score:.2f})")
            score = text_score
            sentiment = text_sentiment

        voice_features = {
            "emotion": raw_emotion,
            "utterance_index": i,
            "full_transcription": full_text,
            "duration_ms": data.get("duration_ms", 0),
            "text_friction_detected": text_score > 0,
        }

        results.append(SentimentResult(
            sentiment=sentiment,
            score=score,
            quote=utt_text,
            timestamp=timestamp,
            chunk_index=i,
            voice_features=voice_features,
        ))

        print(f"[Modulate] Utterance {i} at {timestamp:.1f}s: {sentiment} ({score:.2f}) — \"{utt_text[:60]}\"")

    return results
