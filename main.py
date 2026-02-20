"""FastAPI app: serves the web UI, WebSocket connections, and control API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import logfire
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from auth import SECRET_KEY, send_magic_link, verify_token
from evals import evaluate_assertions, format_beliefs_for_eval
from learning_loop import CHECKPOINT_DIR, FRAMES_DIR, LearningLoop

EVAL_DIR = CHECKPOINT_DIR  # Store eval results alongside checkpoints
from models import WorldState
from scenarios import ALL_SCENARIOS, get_scenario

load_dotenv()
logfire.configure(
    service_name="bayz",
    environment=os.environ.get("RAILWAY_ENVIRONMENT", "development"),
)
logfire.instrument_pydantic_ai()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# -- Singletons ---------------------------------------------------------------

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Per-scenario learning loops (created lazily on first Start)
loops: dict[str, LearningLoop] = {}

# Which scenario the UI is currently viewing (controls frame/state broadcast)
viewed_scenario_id: str = ALL_SCENARIOS[0].id

_broadcast_tasks: list[asyncio.Task] = []


# -- WebSocket connection manager ----------------------------------------------


class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast_bytes(self, data: bytes) -> None:
        dead: list[WebSocket] = []
        for ws in self.active:
            try:
                await ws.send_bytes(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def broadcast_text(self, data: str) -> None:
        dead: list[WebSocket] = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


frame_manager = ConnectionManager()
state_manager = ConnectionManager()


# -- Helpers -------------------------------------------------------------------


def _get_loop(scenario_id: str) -> LearningLoop | None:
    return loops.get(scenario_id)


def _get_scenario_status(scenario_id: str) -> dict:
    """Get status summary for a single scenario. Falls back to checkpoint."""
    loop = loops.get(scenario_id)
    if loop:
        summary = loop.world_state.as_summary_dict()
        summary["scenario_id"] = scenario_id
        summary["status"] = loop.status()
        return summary
    # No running loop — check for saved checkpoint
    ws = LearningLoop.load_checkpoint_state(scenario_id)
    if ws:
        summary = ws.as_summary_dict()
        summary["scenario_id"] = scenario_id
        summary["status"] = "checkpoint"
        return summary
    return {"status": "idle"}


# -- Background broadcast tasks ------------------------------------------------


async def _frame_broadcast_task() -> None:
    """Consume frame_queue from the viewed loop and broadcast to WebSocket clients."""
    while True:
        loop = loops.get(viewed_scenario_id)
        if loop and loop.is_running:
            try:
                frame_png = await asyncio.wait_for(
                    loop.frame_queue.get(), timeout=0.5
                )
                await frame_manager.broadcast_bytes(frame_png)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        else:
            await asyncio.sleep(0.1)


async def _state_broadcast_task() -> None:
    """Consume state_queue from the viewed loop and broadcast to WebSocket clients."""
    while True:
        loop = loops.get(viewed_scenario_id)
        if loop and loop.is_running:
            try:
                state_dict = await asyncio.wait_for(
                    loop.state_queue.get(), timeout=0.5
                )
                msg = json.dumps({"type": "state", "data": state_dict})
                await state_manager.broadcast_text(msg)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        else:
            await asyncio.sleep(0.1)


# -- Lifespan -----------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # No loops created on startup - user starts them manually
    _broadcast_tasks.extend([
        asyncio.ensure_future(_frame_broadcast_task()),
        asyncio.ensure_future(_state_broadcast_task()),
    ])
    log.info("Bayz started - no scenarios running, waiting for user")
    yield
    for loop in loops.values():
        loop.stop()
    for t in _broadcast_tasks:
        t.cancel()
    log.info("Bayz stopped")


# -- App -----------------------------------------------------------------------

app = FastAPI(title="Bayz", lifespan=lifespan)
logfire.instrument_fastapi(app, excluded_urls="/api/status,/api/frames")


# -- Auth middleware -----------------------------------------------------------
# Order matters: SessionMiddleware must be outermost (added last) so it runs
# before the auth check and populates request.session.


_AUTH_ENABLED = bool(os.environ.get("RAILWAY_ENVIRONMENT"))


async def _require_auth(request: Request, call_next):
    """Redirect unauthenticated requests to the login page. Skipped in local dev."""
    if not _AUTH_ENABLED:
        return await call_next(request)
    path = request.url.path
    if path.startswith("/auth") or path == "/health":
        return await call_next(request)
    if not request.session.get("email"):
        return RedirectResponse(url="/auth/login", status_code=302)
    return await call_next(request)


from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402

app.add_middleware(BaseHTTPMiddleware, dispatch=_require_auth)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)


# -- Health --------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


# -- Auth routes ---------------------------------------------------------------


@app.get("/auth/login", response_class=HTMLResponse)
async def auth_login(request: Request, sent: str = ""):
    ctx: dict = {"request": request, "message": None, "message_type": ""}
    if sent:
        ctx["message"] = (
            "Check your inbox — a magic link is on its way. "
            "It expires in 1 hour."
        )
        ctx["message_type"] = "success"
    return templates.TemplateResponse("login.html", ctx)


@app.post("/auth/send", response_class=HTMLResponse)
async def auth_send(request: Request):
    form = await request.form()
    email = str(form.get("email", "")).strip().lower()
    # Always show the same success message to avoid email enumeration
    send_magic_link(email)
    return RedirectResponse(url="/auth/login?sent=1", status_code=303)


@app.get("/auth/verify")
async def auth_verify(request: Request, token: str = ""):
    email = verify_token(token)
    if not email:
        ctx = {
            "request": request,
            "message": "This link is invalid or has expired. Please request a new one.",
            "message_type": "error",
        }
        return templates.TemplateResponse("login.html", ctx)
    request.session["email"] = email
    log.info("User authenticated: %s", email)
    return RedirectResponse(url="/", status_code=303)


@app.post("/auth/logout")
async def auth_logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/auth/login", status_code=303)


# -- Page route ----------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -- Control API ---------------------------------------------------------------


@app.get("/api/scenarios")
async def list_scenarios():
    """List all available scenarios."""
    return [
        {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "entity_types": [
                {"kind": t.kind, "color": list(t.color), "count": t.count}
                for t in s.entity_types
            ],
            "assertions": s.assertions,
        }
        for s in ALL_SCENARIOS
    ]


@app.post("/api/scenario/{scenario_id}/start")
async def start_scenario(scenario_id: str):
    """Create loop if needed and start it. Loads checkpoint if available."""
    config = get_scenario(scenario_id)
    loop = loops.get(scenario_id)
    if loop:
        loop.stop()
    loop = LearningLoop(config)
    if loop.load_checkpoint():
        log.info("Resumed from checkpoint: %s", scenario_id)
    loops[scenario_id] = loop
    loop.start()
    log.info("Started scenario: %s", scenario_id)
    return _get_scenario_status(scenario_id)


@app.post("/api/scenario/{scenario_id}/pause")
async def pause_scenario(scenario_id: str):
    loop = _get_loop(scenario_id)
    if loop:
        loop.pause()
        return _get_scenario_status(scenario_id)
    return {"status": "idle"}


@app.post("/api/scenario/{scenario_id}/resume")
async def resume_scenario(scenario_id: str):
    loop = _get_loop(scenario_id)
    if loop:
        loop.resume()
        return _get_scenario_status(scenario_id)
    return {"status": "idle"}


@app.post("/api/scenario/{scenario_id}/stop")
async def stop_scenario(scenario_id: str):
    loop = _get_loop(scenario_id)
    if loop:
        loop.stop()
        return _get_scenario_status(scenario_id)
    return {"status": "idle"}


@app.post("/api/scenario/{scenario_id}/reset")
async def reset_scenario(scenario_id: str):
    loop = _get_loop(scenario_id)
    if loop:
        loop.pause()
        loop.reset()  # also deletes checkpoint
        return _get_scenario_status(scenario_id)
    # No loop, but maybe stale data — clean it up
    path = CHECKPOINT_DIR / f"{scenario_id}.json"
    if path.exists():
        path.unlink()
    frames_dir = FRAMES_DIR / scenario_id
    if frames_dir.exists():
        import shutil
        shutil.rmtree(frames_dir)
    return {"status": "idle"}


@app.post("/api/view/{scenario_id}")
async def view_scenario(scenario_id: str):
    """Switch which scenario's frames/state are broadcast to the UI."""
    global viewed_scenario_id
    get_scenario(scenario_id)  # validate
    viewed_scenario_id = scenario_id
    log.info("Viewing scenario: %s", scenario_id)
    return _get_scenario_status(scenario_id)


@app.get("/api/status")
async def get_status():
    """Get status for all scenarios (for tab indicators)."""
    result = {}
    for s in ALL_SCENARIOS:
        if s.id in loops:
            result[s.id] = {
                "status": loops[s.id].status(),
                "verdict_count": loops[s.id].world_state.verdict_count,
                "beliefs_count": len(loops[s.id].world_state.beliefs),
            }
        elif LearningLoop.checkpoint_exists(s.id):
            result[s.id] = {
                "status": "checkpoint",
                "verdict_count": 0,
                "beliefs_count": 0,
            }
        else:
            result[s.id] = {
                "status": "idle",
                "verdict_count": 0,
                "beliefs_count": 0,
            }
    return {"viewed": viewed_scenario_id, "scenarios": result}


@app.post("/api/scenario/{scenario_id}/eval")
async def run_eval(scenario_id: str):
    """Evaluate beliefs for a specific scenario against its ground-truth rubric."""
    loop = _get_loop(scenario_id)
    if not loop:
        return {"error": "no_loop", "message": "Scenario has not been started yet."}
    if not loop.world_state.beliefs:
        return {"error": "no_beliefs", "message": "No beliefs to evaluate yet."}
    config = get_scenario(scenario_id)
    beliefs_text = format_beliefs_for_eval(loop.world_state)
    assertion_results = await evaluate_assertions(beliefs_text, config.assertions, scenario_id=scenario_id)

    passed_count = sum(1 for r in assertion_results if r["passed"])
    overall_score = sum(r["score"] for r in assertion_results) / len(assertion_results)

    result = {
        "scenario_id": scenario_id,
        "passed": passed_count == len(assertion_results),
        "score": overall_score,
        "assertions_passed": passed_count,
        "assertions_total": len(assertion_results),
        "assertions": assertion_results,
        "beliefs_evaluated": beliefs_text,
    }

    # Persist eval result to disk so it survives page refresh / reboot
    EVAL_DIR.mkdir(exist_ok=True)
    eval_path = EVAL_DIR / f"{scenario_id}_eval.json"
    eval_path.write_text(json.dumps(result, indent=2))

    return result


@app.get("/api/scenario/{scenario_id}/eval")
async def get_last_eval(scenario_id: str):
    """Return the last saved eval result for this scenario, if any."""
    eval_path = EVAL_DIR / f"{scenario_id}_eval.json"
    if not eval_path.exists():
        return {"eval": None}
    try:
        return json.loads(eval_path.read_text())
    except Exception:
        return {"eval": None}


@app.get("/api/frames/{scenario_id}/{frame_id}.png")
async def get_frame(scenario_id: str, frame_id: int):
    """Serve a saved observation frame as PNG."""
    path = FRAMES_DIR / scenario_id / f"{frame_id}.png"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "not found"})
    return FileResponse(path, media_type="image/png")


# -- WebSocket routes ----------------------------------------------------------


@app.websocket("/ws/frames")
async def ws_frames(ws: WebSocket):
    await frame_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        frame_manager.disconnect(ws)


@app.websocket("/ws/state")
async def ws_state(ws: WebSocket):
    await state_manager.connect(ws)
    # Send current state snapshot for viewed scenario (loop or checkpoint)
    status_data = _get_scenario_status(viewed_scenario_id)
    if status_data.get("status") != "idle":
        current = json.dumps({"type": "state", "data": status_data})
        await ws.send_text(current)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        state_manager.disconnect(ws)


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
