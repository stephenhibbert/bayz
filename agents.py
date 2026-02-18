"""Pydantic AI scientist agent with tools for observing and reasoning about simulations."""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)

from dotenv import load_dotenv
load_dotenv()

from models import (
    Belief,
    BeliefSnapshot,
    Observation,
    WorldState,
)

log = logging.getLogger(__name__)

MODEL = "anthropic:claude-sonnet-4-5-20250929"

# Keep well under the Anthropic 100-image limit to leave headroom.
MAX_CONTEXT_IMAGES = 80


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


# -- History processor: cap images sent to the model -------------------------


def _is_image(obj: object) -> bool:
    return isinstance(obj, BinaryContent) and obj.media_type.startswith("image/")


def _cap_images(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Drop older images so the context stays under MAX_CONTEXT_IMAGES.

    Walks the message list in reverse (newest first) and counts images.
    Once the budget is exhausted, any remaining BinaryContent with an image
    media type is replaced with a short text placeholder so the model still
    sees the frame IDs and text but not the raw pixels.

    Images can appear in:
      - ToolReturnPart.content (list or single value)
      - UserPromptPart.content (sequence — pydantic_ai moves BinaryContent
        from tool returns into a UserPromptPart for the model)
    """
    # First pass (reverse): locate every image and decide which to keep.
    # Each entry: (msg_index, part_index, item_index_in_content | None)
    budget = MAX_CONTEXT_IMAGES
    to_strip: list[tuple[int, int, int | None]] = []

    for mi in range(len(messages) - 1, -1, -1):
        msg = messages[mi]
        if not isinstance(msg, ModelRequest):
            continue
        for pi in range(len(msg.parts) - 1, -1, -1):
            part = msg.parts[pi]

            if isinstance(part, (ToolReturnPart, UserPromptPart)):
                content = part.content
                if isinstance(content, (list, tuple)):
                    for ii in range(len(content) - 1, -1, -1):
                        if _is_image(content[ii]):
                            if budget > 0:
                                budget -= 1
                            else:
                                to_strip.append((mi, pi, ii))
                elif _is_image(content):
                    if budget > 0:
                        budget -= 1
                    else:
                        to_strip.append((mi, pi, None))

    if not to_strip:
        return messages

    # Second pass: shallow-copy affected messages/parts and replace images.
    messages = list(messages)
    copied_msgs: set[int] = set()
    copied_parts: set[tuple[int, int]] = set()
    placeholder = "[image removed from context]"

    for mi, pi, ii in to_strip:
        if mi not in copied_msgs:
            orig = messages[mi]
            assert isinstance(orig, ModelRequest)
            patched = copy.copy(orig)
            patched.parts = list(orig.parts)
            messages[mi] = patched
            copied_msgs.add(mi)

        msg = messages[mi]
        assert isinstance(msg, ModelRequest)

        if (mi, pi) not in copied_parts:
            msg.parts[pi] = copy.copy(msg.parts[pi])
            part = msg.parts[pi]
            if isinstance(part.content, (list, tuple)):
                part.content = list(part.content)
            copied_parts.add((mi, pi))

        part = msg.parts[pi]

        if ii is not None and isinstance(part.content, list):
            part.content[ii] = placeholder
        else:
            part.content = placeholder

    stripped = len(to_strip)
    log.info("History processor: stripped %d old image(s), keeping %d", stripped, MAX_CONTEXT_IMAGES)
    return messages


# -- Agent -------------------------------------------------------------------


scientist_agent = Agent(
    model=MODEL,
    deps_type=AgentDeps,
    output_type=str,
    history_processors=[_cap_images],
    system_prompt="""\
You are an autonomous scientific observer. You have been placed in front of a live
simulation that you know nothing about. No one will tell you what is in it.

Your mission: figure out the rules of this simulation purely through observation.
Discover what entities exist, how they behave, and what interactions govern them.
Form hypotheses, test them with more observations, and update your confidence
based on what you observe.

Use your tools:
- get_frames: capture frames from the live simulation (you can request multiple
  adjacent frames to observe motion and interactions over time)
- record_observation: name and save a bundle of frames as a reusable observation;
  you must do this before citing frames in a hypothesis or belief update
- propose_hypothesis: propose a new rule you believe governs entity interactions,
  citing the observation IDs that support it
- update_belief: revise your confidence in a hypothesis, citing the observation IDs
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
    observation_id: str,
    description: str,
    frame_ids: list[int],
) -> str:
    """Store a named bundle of frames as a reusable observation.

    Call this when you have captured frames that clearly show something
    relevant — e.g. an interaction you want to reference when proposing
    or updating beliefs. The same observation can be cited by multiple
    hypotheses.

    Args:
        observation_id: Unique slug for this observation, e.g. 'obs-001'.
        description: What these frames show, e.g. 'yellow attracts blues over 5 frames'.
        frame_ids: The frame numbers captured (from get_frames output).
    """
    await ctx.deps.paused_event.wait()
    if ctx.deps.stop_event.is_set():
        raise RuntimeError("Agent stopped")

    if ctx.deps.world_state.get_observation(observation_id):
        return f"Observation '{observation_id}' already exists."

    obs = Observation(id=observation_id, description=description, frame_ids=frame_ids)
    ctx.deps.world_state.add_observation(obs)

    # Frames are already saved to disk by get_frames at capture time.
    # We only store the metadata here.
    ctx.deps.push_state_cb()
    log.info("Observation stored: %s — %s (%d frames)", observation_id, description, len(frame_ids))
    return f"Observation '{observation_id}' stored: {description} ({len(frame_ids)} frames)."


@scientist_agent.tool
async def propose_hypothesis(
    ctx: RunContext[AgentDeps],
    hypothesis_id: str,
    subject: str,
    predicate: str,
    target: str,
    description: str,
    observation_ids: list[str],
) -> str:
    """Propose a new hypothesis about an interaction rule in the simulation.

    Args:
        hypothesis_id: Unique slug for this hypothesis, e.g. 'red-chases-blue'.
        subject: The entity type performing the action.
        predicate: The interaction type (e.g. chases, flees, attracts, ignores, feeds_on).
        target: The entity type being acted upon.
        description: Plain English statement of the rule.
        observation_ids: IDs of stored observations that support this proposal.
                         Call record_observation first if you haven't already.
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

    # Validate observation IDs
    unknown = [oid for oid in observation_ids if not ctx.deps.world_state.get_observation(oid)]
    if unknown:
        return f"Unknown observation IDs: {unknown}. Call record_observation first."

    from models import Hypothesis
    hyp = Hypothesis.model_validate({
        "id": hypothesis_id,
        "subject": subject,
        "predicate": predicate,
        "object": target,
        "description": description,
    })

    # Count frames across referenced observations
    initial_frame_count = sum(
        len(ctx.deps.world_state.get_observation(oid).frame_ids)
        for oid in observation_ids
        if ctx.deps.world_state.get_observation(oid)
    )

    belief = Belief(
        hypothesis=hyp,
        prior=0.5,
        likelihood=0.5,
        posterior=0.5,
        observation_ids=list(observation_ids),
        observation_count=initial_frame_count,
    )
    belief.history.append(BeliefSnapshot(
        iteration=ctx.deps.world_state.loop_iteration,
        prior=0.5,
        likelihood=0.5,
        posterior=0.5,
        observation_count=initial_frame_count,
        status="proposed",
        reasoning="Initial proposal.",
        observation_ids=list(observation_ids),
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
    observation_ids: list[str],
) -> str:
    """Revise your confidence in an existing hypothesis based on new observations.

    Args:
        hypothesis_id: The ID of the hypothesis to update.
        posterior: Your updated confidence score (0.0–1.0).
                   Use values above 0.8 only with strong, repeated evidence.
        likelihood: How well the observations match this hypothesis (0.0–1.0).
        reasoning: Brief explanation for why you are changing the score.
        observation_ids: IDs of stored observations used in this update.
                         Call record_observation first if you haven't already.
    """
    await ctx.deps.paused_event.wait()
    if ctx.deps.stop_event.is_set():
        raise RuntimeError("Agent stopped")

    belief = ctx.deps.world_state.get_belief(hypothesis_id)
    if not belief:
        return f"No hypothesis found with id '{hypothesis_id}'. Call propose_hypothesis first."

    # Validate observation IDs
    unknown = [oid for oid in observation_ids if not ctx.deps.world_state.get_observation(oid)]
    if unknown:
        return f"Unknown observation IDs: {unknown}. Call record_observation first."

    belief.prior = belief.posterior
    belief.posterior = max(0.0, min(1.0, posterior))
    belief.likelihood = max(0.0, min(1.0, likelihood))
    belief.last_updated = time.time()

    # Add new observation IDs and count frames
    new_ids = [oid for oid in observation_ids if oid not in belief.observation_ids]
    belief.observation_ids.extend(new_ids)
    new_frame_count = sum(
        len(ctx.deps.world_state.get_observation(oid).frame_ids)
        for oid in new_ids
        if ctx.deps.world_state.get_observation(oid)
    )
    belief.observation_count += new_frame_count

    belief.update_status()

    ctx.deps.world_state.loop_iteration += 1
    ctx.deps.world_state.last_belief_update = time.time()

    belief.history.append(BeliefSnapshot(
        iteration=ctx.deps.world_state.loop_iteration,
        prior=belief.prior,
        likelihood=belief.likelihood,
        posterior=belief.posterior,
        observation_count=belief.observation_count,
        status=belief.status,
        reasoning=reasoning,
        observation_ids=list(observation_ids),
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
        obs_list = ", ".join(b.observation_ids) if b.observation_ids else "none"
        lines.append(
            f"  [{b.hypothesis.id}] {b.hypothesis.subject} {b.hypothesis.predicate} "
            f"{b.hypothesis.object_}: \"{b.hypothesis.description}\" "
            f"(posterior={b.posterior:.2f}, observation_frames={b.observation_count}, "
            f"status={b.status}, observations=[{obs_list}])"
        )
    return "\n".join(lines)
