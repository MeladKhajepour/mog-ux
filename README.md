# mog UX — Autonomous UX Analysis Platform

mog UX is an AI-powered platform that autonomously analyzes user testing sessions to surface friction points and generate actionable design improvements. It uses a multi-agent architecture where specialized agents work independently and in parallel, each responsible for a distinct part of the analysis pipeline.

## Architecture

The system follows an **ACE (Autonomous Cognitive Entity)** architecture with three layers: **Sensing**, **Brain**, and **Output**. Each layer is composed of autonomous agents that operate independently, communicate through an event queue, and contribute to a shared evolving playbook.

```
Video Upload
     |
     v
┌─────────────────────────────────────────────────┐
│  SENSING LAYER                                  │
│                                                 │
│  Modulate Agent ──> Friction Detection          │
│  (voice + text)     (score > 0.6?)              │
│                          |                      │
│                          v                      │
│                    Reka Agent                    │
│                    (visual analysis)             │
│                          |                      │
│                          v                      │
│                    FrictionEvent                 │
└─────────────────────┬───────────────────────────┘
                      |  event queue
                      v
┌─────────────────────────────────────────────────┐
│  BRAIN LAYER                                    │
│                                                 │
│  Gemini Agent (Reflector)                       │
│  diagnose root cause + suggest fix              │
│           |                                     │
│           v                                     │
│  Curate immediately ──> Cards visible           │
│           |                                     │
│     ┌─────┼─────────┐                           │
│     v     v         v        (in parallel)      │
│  Yutori  Nano     Learner                       │
│  Agent   Banana   Agent                         │
│          Agent    (mem0)                         │
└─────────────────────────────────────────────────┘
                      |
                      v
┌─────────────────────────────────────────────────┐
│  OUTPUT LAYER                                   │
│                                                 │
│  Designer's Brief Dashboard                     │
│  - Friction Logs (with problem screenshots)     │
│  - Hard Strategies (with suggested UI mockups)  │
│  - Benchmarked Solutions                        │
│  - Live pipeline status                         │
└─────────────────────────────────────────────────┘
```

## Autonomous Agents

### Modulate Agent — Voice & Text Sentiment Analysis

The Modulate agent uses the **Velma 2 STT Batch API** to perform prosodic analysis on the audio waveform — detecting emotion from pitch, pace, volume, and vocal patterns. It operates autonomously on each audio chunk, extracting utterances with speaker diarization and mapping detected emotions (Frustrated, Confused, Hesitant, etc.) to friction scores.

The agent also runs a **text-based friction analysis** in parallel, scanning the transcript for phrases that indicate confusion or frustration ("can't figure out", "not working", "where is"). This dual-signal approach ensures friction is caught even when a user speaks calmly while expressing frustration through their words. The stronger signal wins.

When the friction score exceeds the threshold (0.6), the agent autonomously triggers frame extraction and hands off to the Reka agent.

### Reka Agent — Visual Context Analysis

The Reka agent uses **Reka Flash**, a multimodal vision model, to autonomously analyze the extracted video frame at the exact moment of detected friction. It receives the screenshot along with contextual data from the Modulate agent (what the user said, their sentiment) and identifies:

- The specific UI element the user is struggling with
- The page or screen they're on
- A description of the visual problem

This agent works independently — it only needs the frame and the acoustic context. Its output becomes the `VisualContext` attached to each `FrictionEvent`, giving the downstream Reflector agent the full picture: what the user felt, what they said, and what they were looking at.

### Gemini Agent (Reflector) — Root Cause Diagnosis

The Reflector agent uses **Gemini 3 Flash** to perform autonomous root cause analysis on each friction event. It receives the combined output of the Modulate and Reka agents and produces a structured diagnosis: the specific UX flaw, its severity, its category, and an actionable fix suggestion.

Before analyzing, it consults the Learner agent's memory to check for recurring patterns from previous sessions. If the same issue has appeared before, it autonomously escalates severity and references the pattern — enabling **continual learning** across sessions.

### Yutori Agent — UX Benchmark Research

The Yutori agent uses the **Yutori Research API** to autonomously search for industry best practices related to each diagnosed issue. It operates in parallel with the mockup and memory agents — it doesn't block the dashboard from showing results.

Given a root cause diagnosis and UX category, it creates a research task that returns:
- The authoritative source or standard
- An actionable UX recommendation
- Real-world examples of apps that handle the issue well

These become "Benchmarked Solutions" cards on the dashboard, giving designers not just what's wrong and how to fix it, but how the best products in the industry have already solved the same problem.

### Nano Banana Agent — UI Mockup Generation

The Nano Banana agent uses **Gemini 3 Pro** (image generation) to autonomously generate a visual mockup showing the suggested fix applied to the original screenshot. It takes the extracted frame, the diagnosed problem, and the suggested fix, and produces a modified version of the screenshot that looks production-ready.

This agent runs in the background after the cards are already visible on the dashboard. A generating indicator is shown while it works, and the mockup appears in-place when ready.

### Learner Agent — Continual Learning with mem0

The Learner agent provides **cross-session memory** using **mem0** with a Qdrant vector store. It operates three learning loops:

1. **Per-event learning**: After each friction event is curated, the agent stores the insight (root cause, severity, category, page, element, user quote) as a searchable memory.

2. **Session summary**: After all chunks from an upload are processed, the agent extracts cross-event patterns — which pages had the most friction, what the dominant sentiment was — and stores a session-level summary.

3. **Recall**: Before the Reflector analyzes a new event, the Learner agent retrieves the top 5 most relevant past memories via semantic search. This allows the system to recognize recurring issues, escalate severity for repeat problems, and build institutional knowledge over time.

This gives the entire system **continual learning** — it gets smarter with each session it analyzes.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```
GEMINI_API_KEY=...
MODULATE_API_KEY=...
REKA_API_KEY=...
YUTORI_API_KEY=...
```

Run the server:

```bash
python main.py
```

Open http://localhost:8000, upload a user testing video, and the autonomous agents will analyze it and populate the Designer's Brief dashboard.

## Testing

```bash
pytest tests/ -v
```
