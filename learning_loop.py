"""Async orchestrator: connects the simulator to the AI agent and web UI."""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
import time
from pathlib import Path

from agents import AgentDeps, scientist_agent
from models import WorldState
from scenarios import ScenarioConfig, PREDATOR_PREY
from simulator import Simulator

log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
FRAMES_DIR = Path(__file__).parent / "frames"

SIM_DT = 1.0 / 30
FRAME_BROADCAST_INTERVAL = 0.1  # 10 fps to UI


class LearningLoop:
    """Coordinates simulator thread, AI scientist agent, and WebSocket queues."""

    def __init__(self, scenario: ScenarioConfig | None = None) -> None:
        self.scenario = scenario or PREDATOR_PREY
        self.simulator = Simulator(config=self.scenario)
        self.world_state = WorldState()
        self.frame_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=5)
        self.state_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=10)
        self._stop_event = threading.Event()
        self._paused = asyncio.Event()
        self._paused.set()  # starts unpaused
        self._tasks: list[asyncio.Task] = []

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()

    def start(self) -> None:
        """Start the simulator thread, frame broadcast, and agent."""
        self._stop_event.clear()
        self._paused.set()
        sim_thread = threading.Thread(target=self._sim_thread, daemon=True)
        sim_thread.start()
        self._tasks = [
            asyncio.ensure_future(self._frame_broadcast_loop()),
            asyncio.ensure_future(self._run_agent()),
        ]

    def stop(self) -> None:
        self._stop_event.set()
        self._paused.set()  # unblock any paused tool calls
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    def pause(self) -> None:
        """Pause the AI agent. Simulator and frame broadcast keep running."""
        self._paused.clear()
        log.info("Learning loop paused")

    def resume(self) -> None:
        """Resume the AI agent."""
        self._paused.set()
        log.info("Learning loop resumed")

    def reset(self) -> None:
        """Reset the simulator and world state. Stays paused after reset."""
        self.simulator = Simulator(config=self.scenario)
        self.world_state = WorldState()
        self.delete_checkpoint()
        self.delete_frames()
        self._push_state()
        log.info("Reset: fresh simulator and world state for %s", self.scenario.id)

    # -- Checkpoints -----------------------------------------------------------

    def _checkpoint_path(self) -> Path:
        return CHECKPOINT_DIR / f"{self.scenario.id}.json"

    def save_checkpoint(self) -> None:
        """Write current world state to a JSON checkpoint file."""
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        path = self._checkpoint_path()
        data = self.world_state.model_dump_json(indent=2)
        path.write_text(data)
        log.info(
            "Checkpoint saved: %s (%d beliefs, iter %d)",
            path.name,
            len(self.world_state.beliefs),
            self.world_state.loop_iteration,
        )

    def load_checkpoint(self) -> bool:
        """Load world state from checkpoint if it exists. Returns True if loaded."""
        path = self._checkpoint_path()
        if not path.exists():
            return False
        try:
            data = path.read_text()
            self.world_state = WorldState.model_validate_json(data)
            log.info(
                "Checkpoint loaded: %s (%d beliefs, iter %d)",
                path.name,
                len(self.world_state.beliefs),
                self.world_state.loop_iteration,
            )
            return True
        except Exception:
            log.exception("Failed to load checkpoint %s", path)
            return False

    def delete_checkpoint(self) -> None:
        """Remove the checkpoint file for this scenario."""
        path = self._checkpoint_path()
        if path.exists():
            path.unlink()
            log.info("Checkpoint deleted: %s", path.name)

    def _save_frame(self, frame_id: int, frame_png: bytes) -> None:
        """Save an observed frame to disk for later preview."""
        d = FRAMES_DIR / self.scenario.id
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{frame_id}.png").write_bytes(frame_png)

    def delete_frames(self) -> None:
        """Remove all saved frames for this scenario."""
        d = FRAMES_DIR / self.scenario.id
        if d.exists():
            shutil.rmtree(d)
            log.info("Frames deleted: %s", self.scenario.id)

    @staticmethod
    def checkpoint_exists(scenario_id: str) -> bool:
        return (CHECKPOINT_DIR / f"{scenario_id}.json").exists()

    @staticmethod
    def load_checkpoint_state(scenario_id: str) -> WorldState | None:
        """Load a checkpoint without creating a full LearningLoop."""
        path = CHECKPOINT_DIR / f"{scenario_id}.json"
        if not path.exists():
            return None
        try:
            return WorldState.model_validate_json(path.read_text())
        except Exception:
            log.exception("Failed to load checkpoint %s", path)
            return None

    def status(self) -> str:
        if self._stop_event.is_set():
            return "stopped"
        if not self._paused.is_set():
            return "paused"
        return "running"

    # -- Simulator thread (sync) -----------------------------------------------

    def _sim_thread(self) -> None:
        while not self._stop_event.is_set():
            self.simulator.tick(SIM_DT)
            time.sleep(SIM_DT)

    # -- Frame broadcast (async, 10 fps) ---------------------------------------

    async def _frame_broadcast_loop(self) -> None:
        while not self._stop_event.is_set():
            frame_png = self.simulator.capture_frame()
            if frame_png:
                try:
                    self.frame_queue.put_nowait(frame_png)
                except asyncio.QueueFull:
                    try:
                        self.frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    self.frame_queue.put_nowait(frame_png)
            await asyncio.sleep(FRAME_BROADCAST_INTERVAL)

    # -- Agent (async) ---------------------------------------------------------

    async def _run_agent(self) -> None:
        """Run the scientist agent as a fully autonomous agentic loop."""
        # Let the simulator warm up before the agent starts observing
        await asyncio.sleep(2.0)

        deps = AgentDeps(
            simulator=self.simulator,
            world_state=self.world_state,
            push_state_cb=self._push_state,
            save_checkpoint_cb=self.save_checkpoint,
            save_frame_cb=self._save_frame,
            paused_event=self._paused,
            stop_event=self._stop_event,
        )

        try:
            result = await scientist_agent.run(
                "Begin observing the simulation. Use your tools to discover what is happening.",
                deps=deps,
            )
            log.info("Agent completed. Summary: %s", result.output)
            self.world_state.last_belief_update = time.time()
            self._push_state()
            self.save_checkpoint()
        except asyncio.CancelledError:
            log.info("Agent task cancelled")
        except Exception:
            log.exception("Agent loop failed")
        finally:
            # Auto-pause so the UI shows the agent has finished
            if not self._stop_event.is_set():
                self.pause()

    # -- State broadcast -------------------------------------------------------

    def _push_state(self) -> None:
        summary = self.world_state.as_summary_dict()
        summary["scenario_id"] = self.scenario.id
        summary["status"] = self.status()
        try:
            self.state_queue.put_nowait(summary)
        except asyncio.QueueFull:
            try:
                self.state_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.state_queue.put_nowait(summary)
