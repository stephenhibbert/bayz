"""Pydantic AI scientist agent with tools for observing and reasoning about simulations."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

from pydantic_ai import Agent, BinaryContent, RunContext

from dotenv import load_dotenv
load_dotenv()

from models import (
    Belief,
    BeliefSnapshot,
    Evidence,
    FrameObservation,
    ObservedEntity,
    WorldState,
)

log = logging.getLogger(__name__)

MODEL = "anthropic:claude-sonnet-4-5-20250929"


# -- Deps --------------------------------------------------------------------


@dataclass
class AgentDeps:
    """Dependencies injected into every tool call."""

    simulator: object  # Simulator (avoid circular import at module level)
    world_state: WorldState
    push_state_cb: Callable[[], None]
    save_checkpoint_cb: Callable[[], None]
    save_frame_cb: Callable[[int, bytes], None]
    paused_event: asyncio.Event
    stop_event: threading.Event


# -- Agent -------------------------------------------------------------------


scientist_agent = Agent(
    model=MODEL,
    deps_type=AgentDeps,
    output_type=str,
    system_prompt="""\
You are an autonomous scientific observer. You have been placed in front of a live
simulation that you know nothing about. No one will tell you what is in it.

Your mission: figure out the rules of this simulation purely through observation.
Discover what entities exist, how they behave, and what interactions govern them.
Form hypotheses, test them with more observations, and update your confidence
based on the evidence you collect.

Use your tools:
- get_frames: capture frames from the live simulation (you can request multiple
  adjacent frames to observe motion and interactions over time)
- record_observation: store what you saw in a frame
- store_evidence: name and save a bundle of frames as a reusable piece of evidence;
  you must do this before citing frames in a hypothesis or belief update
- propose_hypothesis: propose a new rule you believe governs entity interactions,
  citing the evidence IDs that support it
- update_belief: revise your confidence in a hypothesis, citing the evidence IDs
  that informed the update
- get_current_beliefs: review your existing hypotheses and their confidence scores

Work like a scientist: observe first, then hypothesise, then gather more evidence.
Importantly, propose your first hypotheses early — after just 2-3 observations —
even if you are uncertain. Early hypotheses at low confidence (0.3–0.5) are fine;
you will refine them with more evidence. Do not wait until you are sure.
When you are genuinely confident you understand the key rules of this simulation,
return a concise summary of your conclusions.""",
)


# -- Tools -------------------------------------------------------------------


@scientist_agent.tool
async def get_frames(
    ctx: RunContext[AgentDeps],
    n: int = 1,
    delay_secs: float = 1.0,
) -> list:
    """Capture n consecutive frames from the live simulation.

    Args:
        n: Number of frames to capture (1-8).
        delay_secs: Seconds to wait between each capture (default 1.0).

    Each frame is returned as an image you can directly examine. Request more
    frames with a longer delay to observe motion patterns and interactions.
    """
    await ctx.deps.paused_event.wait()
    if ctx.deps.stop_event.is_set():
        raise RuntimeError("Agent stopped")

    n = max(1, min(n, 8))
    results = []

    for i in range(n):
        if i > 0:
            await asyncio.sleep(delay_secs)

        frame_data: bytes = ctx.deps.simulator.capture_frame()
        frame_id: int = ctx.deps.simulator.tick_count

        ctx.deps.save_frame_cb(frame_id, frame_data)
        ctx.deps.world_state.total_frames_observed += 1

        results.append(f"Frame #{frame_id}:")
        results.append(BinaryContent(data=frame_data, media_type="image/png"))

    ctx.deps.push_state_cb()
    return results


@scientist_agent.tool
async def record_observation(
    ctx: RunContext[AgentDeps],
    frame_id: int,
    scene_summary: str,
    entities: list[ObservedEntity],
) -> str:
    """Store your observations from a frame you have examined.

    Args:
        frame_id: The frame number (shown in the get_frames output).
        scene_summary: A one-sentence description of what you see overall.
        entities: The individual entities you identified in this frame.
    """
    await ctx.deps.paused_event.wait()
    if ctx.deps.stop_event.is_set():
        raise RuntimeError("Agent stopped")

    obs = FrameObservation(
        frame_id=frame_id,
        entities=entities,
        scene_summary=scene_summary,
    )
    ctx.deps.world_state.recent_observations.append(obs)
    if len(ctx.deps.world_state.recent_observations) > 20:
        ctx.deps.world_state.recent_observations.pop(0)

    ctx.deps.push_state_cb()
    log.info(
        "Observation recorded: frame %d (%d entities): %s",
        frame_id,
        len(entities),
        scene_summary,
    )
    return f"Observation stored: frame {frame_id}, {len(entities)} entities."


@scientist_agent.tool
async def store_evidence(
    ctx: RunContext[AgentDeps],
    evidence_id: str,
    description: str,
    frame_ids: list[int],
) -> str:
    """Store a named bundle of frames as a reusable piece of evidence.

    Call this when you have captured frames that clearly show something
    relevant — e.g. an interaction you want to reference when proposing
    or updating beliefs. The same evidence bundle can be cited by multiple
    hypotheses.

    Args:
        evidence_id: Unique slug for this evidence bundle, e.g. 'ev-001'.
        description: What these frames show, e.g. 'yellow attracts blues over 5 frames'.
        frame_ids: The frame numbers captured (from get_frames output).
    """
    await ctx.deps.paused_event.wait()
    if ctx.deps.stop_event.is_set():
        raise RuntimeError("Agent stopped")

    if ctx.deps.world_state.get_evidence(evidence_id):
        return f"Evidence '{evidence_id}' already exists."

    ev = Evidence(id=evidence_id, description=description, frame_ids=frame_ids)
    ctx.deps.world_state.add_evidence(ev)

    # Frames are already saved to disk by get_frames at capture time.
    # We only store the metadata here.
    ctx.deps.push_state_cb()
    log.info("Evidence stored: %s — %s (%d frames)", evidence_id, description, len(frame_ids))
    return f"Evidence '{evidence_id}' stored: {description} ({len(frame_ids)} frames)."


@scientist_agent.tool
async def propose_hypothesis(
    ctx: RunContext[AgentDeps],
    hypothesis_id: str,
    subject: str,
    predicate: str,
    target: str,
    description: str,
    evidence_ids: list[str],
) -> str:
    """Propose a new hypothesis about an interaction rule in the simulation.

    Args:
        hypothesis_id: Unique slug for this hypothesis, e.g. 'red-chases-blue'.
        subject: The entity type performing the action.
        predicate: The interaction type (e.g. chases, flees, attracts, ignores, feeds_on).
        target: The entity type being acted upon.
        description: Plain English statement of the rule.
        evidence_ids: IDs of stored evidence bundles that support this proposal.
                      Call store_evidence first if you haven't already.
    """
    await ctx.deps.paused_event.wait()
    if ctx.deps.stop_event.is_set():
        raise RuntimeError("Agent stopped")

    # Reject exact ID duplicates
    if ctx.deps.world_state.get_belief(hypothesis_id):
        return f"Hypothesis '{hypothesis_id}' already exists. Use update_belief to revise it."

    # Reject semantic duplicates by (subject, predicate, target) triple
    triple = (subject, predicate, target)
    for b in ctx.deps.world_state.beliefs:
        if (b.hypothesis.subject, b.hypothesis.predicate, b.hypothesis.object_) == triple:
            return (
                f"A semantically equivalent hypothesis already exists: '{b.hypothesis.id}'. "
                f"Use update_belief to revise it."
            )

    # Validate evidence IDs
    unknown = [eid for eid in evidence_ids if not ctx.deps.world_state.get_evidence(eid)]
    if unknown:
        return f"Unknown evidence IDs: {unknown}. Call store_evidence first."

    from models import Hypothesis
    hyp = Hypothesis.model_validate({
        "id": hypothesis_id,
        "subject": subject,
        "predicate": predicate,
        "object": target,
        "description": description,
    })

    # Count frames across referenced evidence
    initial_frame_count = sum(
        len(ctx.deps.world_state.get_evidence(eid).frame_ids)
        for eid in evidence_ids
        if ctx.deps.world_state.get_evidence(eid)
    )

    belief = Belief(
        hypothesis=hyp,
        prior=0.5,
        likelihood=0.5,
        posterior=0.5,
        evidence_ids=list(evidence_ids),
        evidence_count=initial_frame_count,
    )
    belief.history.append(BeliefSnapshot(
        iteration=ctx.deps.world_state.loop_iteration,
        prior=0.5,
        likelihood=0.5,
        posterior=0.5,
        evidence_count=initial_frame_count,
        status="proposed",
        reasoning="Initial proposal.",
        evidence_ids=list(evidence_ids),
    ))

    ctx.deps.world_state.upsert_belief(belief)
    ctx.deps.push_state_cb()
    ctx.deps.save_checkpoint_cb()
    log.info("Hypothesis proposed: %s — %s", hypothesis_id, description)
    return f"Hypothesis '{hypothesis_id}' added with initial confidence 0.50."


@scientist_agent.tool
async def update_belief(
    ctx: RunContext[AgentDeps],
    hypothesis_id: str,
    posterior: float,
    likelihood: float,
    reasoning: str,
    evidence_ids: list[str],
) -> str:
    """Revise your confidence in an existing hypothesis based on new evidence.

    Args:
        hypothesis_id: The ID of the hypothesis to update.
        posterior: Your updated confidence score (0.0–1.0).
                   Use values above 0.8 only with strong, repeated evidence.
        likelihood: How well the observations match this hypothesis (0.0–1.0).
        reasoning: Brief explanation for why you are changing the score.
        evidence_ids: IDs of stored evidence bundles used in this update.
                      Call store_evidence first if you haven't already.
    """
    await ctx.deps.paused_event.wait()
    if ctx.deps.stop_event.is_set():
        raise RuntimeError("Agent stopped")

    belief = ctx.deps.world_state.get_belief(hypothesis_id)
    if not belief:
        return f"No hypothesis found with id '{hypothesis_id}'. Call propose_hypothesis first."

    # Validate evidence IDs
    unknown = [eid for eid in evidence_ids if not ctx.deps.world_state.get_evidence(eid)]
    if unknown:
        return f"Unknown evidence IDs: {unknown}. Call store_evidence first."

    belief.prior = belief.posterior
    belief.posterior = max(0.0, min(1.0, posterior))
    belief.likelihood = max(0.0, min(1.0, likelihood))
    belief.last_updated = time.time()

    # Add new evidence IDs and count frames
    new_ids = [eid for eid in evidence_ids if eid not in belief.evidence_ids]
    belief.evidence_ids.extend(new_ids)
    new_frame_count = sum(
        len(ctx.deps.world_state.get_evidence(eid).frame_ids)
        for eid in new_ids
        if ctx.deps.world_state.get_evidence(eid)
    )
    belief.evidence_count += new_frame_count

    belief.update_status()

    ctx.deps.world_state.loop_iteration += 1
    ctx.deps.world_state.last_belief_update = time.time()

    belief.history.append(BeliefSnapshot(
        iteration=ctx.deps.world_state.loop_iteration,
        prior=belief.prior,
        likelihood=belief.likelihood,
        posterior=belief.posterior,
        evidence_count=belief.evidence_count,
        status=belief.status,
        reasoning=reasoning,
        evidence_ids=list(evidence_ids),
    ))

    ctx.deps.world_state.upsert_belief(belief)
    ctx.deps.push_state_cb()
    ctx.deps.save_checkpoint_cb()
    log.info(
        "Belief updated: %s → posterior=%.2f, status=%s",
        hypothesis_id,
        belief.posterior,
        belief.status,
    )
    return (
        f"Belief '{hypothesis_id}' updated: posterior={belief.posterior:.2f}, "
        f"status={belief.status}."
    )


@scientist_agent.tool
async def get_current_beliefs(ctx: RunContext[AgentDeps]) -> str:
    """Review all current hypotheses and their confidence scores.

    Call this to remind yourself of what you have already proposed before
    deciding whether to add new hypotheses or revise existing ones.
    """
    beliefs = ctx.deps.world_state.beliefs
    if not beliefs:
        return "No hypotheses proposed yet."

    lines = [f"Current beliefs ({len(beliefs)} total, sorted by confidence):"]
    for b in sorted(beliefs, key=lambda b: b.posterior, reverse=True):
        ev_list = ", ".join(b.evidence_ids) if b.evidence_ids else "none"
        lines.append(
            f"  [{b.hypothesis.id}] {b.hypothesis.subject} {b.hypothesis.predicate} "
            f"{b.hypothesis.object_}: \"{b.hypothesis.description}\" "
            f"(posterior={b.posterior:.2f}, evidence_frames={b.evidence_count}, "
            f"status={b.status}, evidence=[{ev_list}])"
        )
    return "\n".join(lines)
