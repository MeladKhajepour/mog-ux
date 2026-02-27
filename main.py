import asyncio
import os
import shutil
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from curator import curate
from generator import process_upload
from learner import store_insight, get_all_memories
from mockup_generator import generate_mockup
from models import FrictionEvent
from playbook import load_playbook
from progress import publish, subscribe
from reflector import reflect_with_benchmarks

# --- Uploads directory ---

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Background pipeline ---

event_queue: asyncio.Queue[FrictionEvent] = asyncio.Queue()


async def brain_pipeline():
    """Background task: pull friction events, reflect, curate."""
    while True:
        event = await event_queue.get()
        try:
            print(f"[Brain] Processing event {event.event_id}...")
            publish("reflecting", "Brain analyzing event...")
            insight, benchmarks = await reflect_with_benchmarks(event)

            # Generate mockup if we have a frame
            frame_url = ""
            mockup_url = ""
            if event.frame_path and os.path.exists(event.frame_path):
                # Build URL from the file path (relative to uploads/)
                rel_path = os.path.relpath(event.frame_path, UPLOAD_DIR)
                frame_url = f"/uploads/{rel_path}"
                try:
                    publish("mockup", "Generating UI mockup via Nano Banana Pro...")
                    mockup_path = await generate_mockup(
                        event.frame_path, insight.root_cause, insight.suggested_fix
                    )
                    if os.path.exists(mockup_path):
                        mockup_rel = os.path.relpath(mockup_path, UPLOAD_DIR)
                        mockup_url = f"/uploads/{mockup_rel}"
                        publish("mockup_done", "Mockup generated")
                except Exception as mockup_err:
                    print(f"[Brain] Mockup generation failed: {mockup_err}")

            publish("curating", f"Curated: {insight.severity} {insight.category}")
            curate(insight, benchmarks, frame_url=frame_url, mockup_url=mockup_url)
            try:
                store_insight(insight)
                publish("learning", "Stored insight in memory")
            except Exception as mem_err:
                print(f"[Brain] Failed to store insight in mem0: {mem_err}")
            print(f"[Brain] Done with event {event.event_id} — {insight.severity} {insight.category}")
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

app = FastAPI(title="Lumina UX — Brain Module", lifespan=lifespan)
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
async def upload_video(request: Request, file: UploadFile = File(...)):
    """Accept a video upload and kick off the sensing pipeline."""
    # Save uploaded file
    filename = file.filename or "upload.mp4"
    video_path = os.path.join(UPLOAD_DIR, filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    size_bytes = os.path.getsize(video_path)
    print(f"[Upload] Saved {filename} ({size_bytes} bytes)")

    # Kick off sensing pipeline in background
    asyncio.create_task(process_upload(video_path, event_queue))

    # If called via fetch (JS), return JSON; otherwise render template
    accept = request.headers.get("accept", "")
    if "application/json" in accept:
        return {"status": "ok", "filename": filename}

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": f"Upload received: {filename}. Processing started — check the dashboard for results.",
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
