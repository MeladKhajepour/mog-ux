import asyncio
import os
import shutil
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request, UploadFile, File
from typing import List
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from curator import curate_friction_log, curate_strategy
from generator import process_upload
from learner import store_insight, get_all_memories, delete_memory, delete_all_memories
from mockup_generator import generate_mockup
from models import FrictionEvent
from playbook import load_playbook, update_mockup_url
from progress import publish, subscribe
from reflector import diagnose, suggest_fix
from yutori_client import search_benchmarks

# --- Uploads directory ---

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Background pipeline ---

event_queue: asyncio.Queue[FrictionEvent] = asyncio.Queue()


async def brain_pipeline():
    """Background task: pull friction events, diagnose → Yutori → fix → curate."""
    while True:
        event = await event_queue.get()
        try:
            print(f"[Brain] Processing event {event.event_id}...")

            # Phase 1: Diagnose (Gemini, fast — no fix yet)
            publish("reflecting", "Diagnosing friction event...")
            partial_insight = await diagnose(event)

            # Build frame URL (needed for both curation phases)
            frame_url = ""
            if event.frame_path and os.path.exists(event.frame_path):
                rel_path = os.path.relpath(event.frame_path, UPLOAD_DIR)
                frame_url = f"/uploads/{rel_path}"

            # Show friction_log card immediately — problem is known
            curate_friction_log(partial_insight, frame_url)
            publish("curating", f"Problem identified: {partial_insight.severity} {partial_insight.category}")
            print(f"[Brain] Friction log live for event {event.event_id}")

            # Phase 2: Yutori research — informed by diagnosis
            publish("researching", "Searching industry benchmarks...")
            benchmarks = await search_benchmarks(partial_insight.root_cause, partial_insight.category)

            # Phase 3: Generate fix, now grounded in research
            publish("suggesting_fix", "Generating research-informed fix...")
            insight = await suggest_fix(partial_insight, benchmarks)

            # Show hard_strategy + benchmark cards together
            curate_strategy(insight, benchmarks, frame_url)
            publish("curating", "Strategy + benchmark live")
            print(f"[Brain] Strategy live for event {event.event_id} — {insight.severity} {insight.category}")

            # Phase 4: Mockup + memory in parallel
            async def do_mockup():
                if not frame_url or not event.frame_path:
                    return
                try:
                    publish("mockup", "Generating suggested UI mockup...")
                    mockup_path = await generate_mockup(
                        event.frame_path, insight.root_cause, insight.suggested_fix
                    )
                    if os.path.exists(mockup_path):
                        mockup_rel = os.path.relpath(mockup_path, UPLOAD_DIR)
                        mockup_url = f"/uploads/{mockup_rel}"
                        update_mockup_url(frame_url, mockup_url)
                        publish("mockup_done", "Suggested UI mockup ready")
                        print(f"[Brain] Mockup generated: {mockup_url}")
                except Exception as e:
                    print(f"[Brain] Mockup generation failed: {e}")
                    publish("mockup_failed", "Mockup generation failed")

            async def do_memory():
                try:
                    await store_insight(insight)
                    publish("learning", "Stored insight in memory")
                except Exception as e:
                    print(f"[Brain] Failed to store insight in mem0: {e}")

            await asyncio.gather(do_mockup(), do_memory())
            print(f"[Brain] Fully done with event {event.event_id}")

        except Exception as e:
            print(f"[Brain] Error processing event {event.event_id}: {e}")
        finally:
            event_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(brain_pipeline())
    yield
    task.cancel()


# --- FastAPI app ---

app = FastAPI(title="mog UX — Brain Module", lifespan=lifespan)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    """Landing page with video upload form."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/progress")
async def progress_stream():
    """SSE endpoint for real-time pipeline progress."""
    return StreamingResponse(
        subscribe(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/upload")
async def upload_video(request: Request, files: List[UploadFile] = File(...)):
    """Accept one or more video uploads and process them sequentially."""
    video_paths: list[tuple[str, str]] = []

    for file in files:
        filename = file.filename or "upload.mp4"
        video_path = os.path.join(UPLOAD_DIR, filename)
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        size_bytes = os.path.getsize(video_path)
        print(f"[Upload] Saved {filename} ({size_bytes} bytes)")
        video_paths.append((filename, video_path))

    total = len(video_paths)

    async def process_all():
        for i, (filename, video_path) in enumerate(video_paths):
            publish("video_start", f"Starting video {i + 1} of {total}: {filename}", str(total))
            await process_upload(video_path, event_queue)

    asyncio.create_task(process_all())

    accept = request.headers.get("accept", "")
    if "application/json" in accept:
        return {"status": "ok", "count": total}

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": f"{total} video(s) queued for analysis.",
    })


@app.post("/events")
async def ingest_event(event: FrictionEvent):
    """Receive a Friction Event from the Sensing module."""
    await event_queue.put(event)
    return {"status": "queued", "event_id": event.event_id, "queue_size": event_queue.qsize()}


@app.get("/playbook")
async def get_playbook():
    """Return the current playbook as JSON."""
    return load_playbook().model_dump()


@app.post("/playbook/clear")
async def clear_playbook():
    """Clear the playbook — removes all bullets."""
    from playbook import save_playbook
    from models import Playbook
    save_playbook(Playbook(session_id="default", bullets=[], last_updated=""))
    return {"status": "cleared"}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the Designer's Brief dashboard."""
    playbook = load_playbook()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "playbook": playbook,
    })


@app.get("/memories", response_class=HTMLResponse)
async def memories_page(request: Request):
    """Render the mem0 memories viewer."""
    return templates.TemplateResponse("memories.html", {"request": request})


@app.get("/api/memories")
async def api_memories():
    """Return all stored mem0 memories as JSON."""
    return get_all_memories()


@app.delete("/api/memories")
async def api_delete_all_memories():
    """Delete all mem0 memories."""
    delete_all_memories()
    return {"status": "cleared"}


@app.delete("/api/memories/{memory_id}")
async def api_delete_memory(memory_id: str):
    """Delete a single mem0 memory by ID."""
    delete_memory(memory_id)
    return {"status": "deleted", "id": memory_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
