"""Fieldwise hub server. Phones on the local Wi-Fi open http://<hub-ip>:8000.

    uvicorn app.server:app --host 0.0.0.0 --port 8000

Nothing here needs the internet except Spark escalations and SAM severity,
which degrade to "queued" / "needs a connection" when offline.
"""
from __future__ import annotations

import socket
import sys
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fieldwise import connectivity  # noqa: E402
from fieldwise.config import settings  # noqa: E402
from fieldwise.labels import CROPS, LABELS  # noqa: E402
from fieldwise.pipeline import Fieldwise  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 15 * 1024 * 1024

app = FastAPI(title="Fieldwise hub")
fw = Fieldwise()


class AskIn(BaseModel):
    question: str
    crop: str | None = None


class OfflineIn(BaseModel):
    forced: bool


def _lan_ip() -> str | None:
    """This laptop's address on the local Wi-Fi (what phones type in). No packets are sent."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.255.255", 1))
            ip = s.getsockname()[0]
            return None if ip.startswith("127.") else ip
    except OSError:
        return None


def _port() -> str:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--port" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--port="):
            return a.split("=", 1)[1]
    return "8000"


@app.on_event("startup")
def _say_where() -> None:
    port, ip = _port(), _lan_ip()
    print(f"\n  Fieldwise hub is up. Open http://localhost:{port}/ on this laptop", flush=True)
    if ip:
        print(f"  Phones on the same Wi-Fi: http://{ip}:{port}/", flush=True)
    print(f"  Guide passages indexed: {len(fw.index)}"
          + ("  (none: run python scripts/ingest_guides.py)" if not len(fw.index) else ""), flush=True)
    ok, detail = fw.glimmer_health()
    print(f"  Muse Glimmer: {detail}", flush=True)
    if fw.replay is not None:
        print("  ! FIELDWISE_GLIMMER_REPLAY=1: Glimmer answers are replays of the benchmark run, labelled in the app. "
              "Say so in the video.", flush=True)
    m = settings.glimmer_model
    if m.endswith(".gguf") or "/" in m or "\\" in m:
        print(f"  ! GLIMMER_MODEL={m} looks like a file path. GLIMMER_MODEL is the name the server answers to "
              "(muse-glimmer); the model file goes in GLIMMER_GGUF.", flush=True)
    print(f"  Muse Spark: {'key found' if settings.meta_api_key else 'no MODEL_API_KEY in .env, so no cloud answers'}\n",
          flush=True)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/index.html", include_in_schema=False)
@app.get("/static/index.html", include_in_schema=False)
def index_alias():
    return RedirectResponse("/")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)    # no icon; stops a harmless 404 line in the log


@app.get("/api/status")
def status():
    return {
        "online": connectivity.is_online(ttl_s=3),
        "forced_offline": connectivity.forced_offline(),
        **dict(zip(("glimmer_up", "glimmer_detail"), fw.glimmer_health())),
        "glimmer_replay": fw.replay is not None,
        "guide_chunks": len(fw.index),
        "qa_threshold": settings.qa_threshold,
        "dx_threshold": settings.dx_threshold,
        "queue_pending": len(fw.queue.pending(limit=500)),
        "crops": list(CROPS),
        "models": {"local": settings.glimmer_model, "cloud": settings.spark_model, "segmentation": settings.sam_model},
    }


@app.post("/api/offline")
def set_offline(body: OfflineIn):
    connectivity.set_forced_offline(body.forced)
    return status()


@app.post("/api/ask")
def ask(body: AskIn):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Type a question first.")
    if body.crop and body.crop not in CROPS:
        raise HTTPException(400, f"Unknown crop. Choose one of: {', '.join(CROPS)}.")
    return asdict(fw.ask(q, body.crop))


async def _read_image(image: UploadFile) -> bytes:
    data = await image.read()
    if not data:
        raise HTTPException(400, "The photo is empty. Take it again.")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "The photo is larger than 15 MB. Take it again at a lower resolution.")
    return data


@app.post("/api/diagnose")
async def diagnose(image: UploadFile = File(...), crop: str = Form(...), note: str = Form("")):
    if crop not in CROPS:
        raise HTTPException(400, f"Unknown crop. Choose one of: {', '.join(CROPS)}.")
    data = await _read_image(image)
    return asdict(fw.diagnose(data, crop, note or None))


@app.post("/api/severity")
async def severity(image: UploadFile = File(...), label: str = Form("")):
    data = await _read_image(image)
    if label and label not in LABELS:
        raise HTTPException(400, "Unknown diagnosis label.")
    return fw.severity(data, label or None)


@app.get("/api/queue")
def queue():
    return {"items": fw.queue.all(limit=30)}


@app.post("/api/queue/flush")
def flush():
    if not connectivity.is_online(ttl_s=0):
        return {"sent": [], "message": "Still offline. Questions stay queued until there's a connection."}
    return {"sent": fw.flush_queue()}
