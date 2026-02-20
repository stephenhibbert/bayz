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
    Hypothesis,
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
    model_settings={"temperature": 0.8},
    system_prompt="""\
You are an autonomous scientific observer watching a live visual feed you know
nothing about. Your mission: discover the rules governing what you see, purely
through observation and reasoning.

Cycle:
1. get_frames — capture frames. Vary your strategy each time: quick burst
   (6-8 frames, 0.3s) to catch fast events, slow sweep (2-3 frames, 3-5s) to
   see large-scale patterns, or medium pace (4-5 frames, 1s) for general
   observation. Don't repeat the same sampling.
2. record_observation — name and save what you saw, listing entities present.
   IMPORTANT: reuse entity names from the known list to avoid synonyms.
3. propose_hypothesis — propose rules between entity types, citing supporting
   observations. Hypothesise early and boldly — even speculative guesses.
   Wrong hypotheses get corrected; missing ones never do.
   - Propose MULTIPLE hypotheses per cycle when you see ambiguous behaviour.
   - For the same entity pair, propose COMPETING hypotheses (e.g. "red chases
     cyan" AND "red is attracted to yellow near cyan") so evidence can
     differentiate them.
   - Prioritise entity pairs with no existing hypothesis (check get_current_beliefs).
4. judge_belief / batch_judge — After recording an observation, judge EVERY
   existing hypothesis that involves the entities you just observed. Use
   batch_judge to update multiple beliefs at once. This is correct Bayesian
   updating: one observation updates all relevant posteriors simultaneously.
5. get_current_beliefs — review hypotheses, confidence scores, and exploration
   guidance showing which pairs need more evidence.

Judging rules:
- "supports" = the observation CLEARLY shows the described behaviour.
  Proximity alone is not enough — you need directional motion or an event.
- "neutral" = you cannot clearly see the behaviour. This is the DEFAULT.
  Most observations are only relevant to a few hypotheses.
- "contradicts" = the observation shows behaviour INCONSISTENT with the
  hypothesis. Don't be afraid to contradict.

Default prior — "no interaction":
- Assume entity pairs DO NOT interact unless you see clear evidence otherwise.
- NEVER propose "X ignores Y" hypotheses. "Ignores" is the default state, not
  a discovery. Only propose hypotheses for ACTIVE interactions you observe:
  chasing, fleeing, attracting, repelling, transforming, consuming, etc.
- This saves your verdict budget for testing real interactions.

Confound awareness:
- When multiple entities move in the same direction, consider whether there is
  a shared cause (e.g. an attractor pulling both) versus a direct pairwise
  interaction (e.g. one chasing the other).
- To disentangle: look for moments when the potential confounder is far away
  or absent. If entity A still approaches entity B when the suspected attractor
  is distant, that is evidence for A-chases-B independent of the attractor.
- Also look at RELATIVE motion between two entities, not just absolute
  direction. If A closes the gap on B while both drift toward C, A may be
  chasing B ON TOP OF a shared attraction to C.
- Propose competing hypotheses for ambiguous observations and let evidence
  differentiate them over time.

After judging, check what is MISSING. Are there entity pairs with no
hypothesis? Behaviours you haven't tested? Propose hypotheses to fill gaps.

When confident, return a concise summary of your conclusions.""",
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
        delay_secs: Seconds to wait between each capture (0.2-5.0).

    Vary your sampling strategy each time:
    - Quick burst (6-8 frames, 0.3s) to catch fast interactions like chasing
    - Slow sweep (2-3 frames, 3-5s) to see large-scale motion and repulsion
    - Medium pace (4-5 frames, 1s) for general observation
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
    entities: list[str],
) -> str:
    """Store a named bundle of frames as a reusable observation.

    Call this when you have captured frames that clearly show something
    relevant. The same observation can be cited by multiple hypotheses.

    Args:
        observation_id: Unique slug for this observation, e.g. 'obs-001'.
        description: What these frames show, e.g. 'red dots chasing cyan dots'.
        frame_ids: The frame numbers captured (from get_frames output).
        entities: Entity names visible in these frames. Reuse names from the
                  known entity list to avoid duplicates.
    """
    await ctx.deps.paused_event.wait()
    if ctx.deps.stop_event.is_set():
        raise RuntimeError("Agent stopped")

    if ctx.deps.world_state.get_observation(observation_id):
        return f"Observation '{observation_id}' already exists."

    # Register any new entity names
    for name in entities:
        ctx.deps.world_state.add_entity(name)

    obs = Observation(
        id=observation_id,
        description=description,
        frame_ids=frame_ids,
        entities=entities,
    )
    ctx.deps.world_state.add_observation(obs)

    ctx.deps.push_state_cb()
    known = ", ".join(ctx.deps.world_state.entities)
    log.info("Observation stored: %s — %s (%d frames)", observation_id, description, len(frame_ids))
    return (
        f"Observation '{observation_id}' stored: {description} ({len(frame_ids)} frames). "
        f"Known entities: [{known}]"
    )


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
        hypothesis_id: Unique slug for this hypothesis, e.g. 'red-chases-cyan'.
        subject: The entity type performing the action.
        predicate: The interaction type (e.g. chases, flees, attracts, repels, ignores).
        target: The entity type being acted upon.
        description: Plain English statement of the rule.
        observation_ids: IDs of stored observations that support this proposal.
                         Call record_observation first if you haven't already.
    """
    await ctx.deps.paused_event.wait()
    if ctx.deps.stop_event.is_set():
        raise RuntimeError("Agent stopped")

    # Reject "ignores" hypotheses — non-interaction is the default prior
    if predicate.lower() in ("ignores", "no_interaction", "does_not_interact", "no interaction"):
        return (
            "No need to propose 'ignores' hypotheses — non-interaction is the "
            "default assumption. Only propose hypotheses for active interactions "
            "you observe (chases, flees, attracts, repels, transforms, etc.)."
        )

    # Reject exact ID duplicates
    if ctx.deps.world_state.get_belief(hypothesis_id):
        return f"Hypothesis '{hypothesis_id}' already exists. Use judge_belief to update it."

    # Reject semantic duplicates by (subject, predicate, target) triple
    triple = (subject, predicate, target)
    for b in ctx.deps.world_state.beliefs:
        if (b.hypothesis.subject, b.hypothesis.predicate, b.hypothesis.object_) == triple:
            return (
                f"A semantically equivalent hypothesis already exists: '{b.hypothesis.id}'. "
                f"Use judge_belief to update it."
            )

    # Validate observation IDs
    unknown = [oid for oid in observation_ids if not ctx.deps.world_state.get_observation(oid)]
    if unknown:
        return f"Unknown observation IDs: {unknown}. Call record_observation first."

    hyp = Hypothesis.model_validate({
        "id": hypothesis_id,
        "subject": subject,
        "predicate": predicate,
        "object": target,
        "description": description,
    })

    belief = Belief(hypothesis=hyp)
    ctx.deps.world_state.upsert_belief(belief)
    ctx.deps.push_state_cb()
    ctx.deps.save_checkpoint_cb()
    log.info("Hypothesis proposed: %s — %s", hypothesis_id, description)
    return (
        f"Hypothesis '{hypothesis_id}' added (confidence {belief.confidence:.0%}). "
        f"Use judge_belief to support or contradict it with evidence."
    )


@scientist_agent.tool
async def judge_belief(
    ctx: RunContext[AgentDeps],
    hypothesis_id: str,
    verdict: str,
    reasoning: str,
    observation_ids: list[str],
) -> str:
    """Judge whether new observations support or contradict an existing hypothesis.

    Args:
        hypothesis_id: The ID of the hypothesis to judge.
        verdict: One of "supports", "contradicts", or "neutral".
        reasoning: Brief explanation for your verdict.
        observation_ids: IDs of stored observations informing this judgment.
                         Call record_observation first if you haven't already.
    """
    await ctx.deps.paused_event.wait()
    if ctx.deps.stop_event.is_set():
        raise RuntimeError("Agent stopped")

    belief = ctx.deps.world_state.get_belief(hypothesis_id)
    if not belief:
        return f"No hypothesis found with id '{hypothesis_id}'. Call propose_hypothesis first."

    # Validate verdict
    if verdict not in ("supports", "contradicts", "neutral"):
        return f"Invalid verdict '{verdict}'. Must be 'supports', 'contradicts', or 'neutral'."

    # Validate observation IDs
    unknown = [oid for oid in observation_ids if not ctx.deps.world_state.get_observation(oid)]
    if unknown:
        return f"Unknown observation IDs: {unknown}. Call record_observation first."

    # Apply verdict
    if verdict == "supports":
        belief.supports += 1
    elif verdict == "contradicts":
        belief.contradicts += 1
    # neutral: no counter change

    belief.last_updated = time.time()
    belief.update_status()

    ctx.deps.world_state.verdict_count += 1

    belief.history.append(BeliefSnapshot(
        iteration=ctx.deps.world_state.verdict_count,
        verdict=verdict,
        reasoning=reasoning,
        observation_ids=list(observation_ids),
    ))

    ctx.deps.world_state.upsert_belief(belief)
    ctx.deps.push_state_cb()
    ctx.deps.save_checkpoint_cb()

    symbol = {"supports": "+", "contradicts": "-", "neutral": "~"}[verdict]
    log.info(
        "Belief judged: %s [%s] → confidence=%.0f%%, status=%s",
        hypothesis_id, symbol, belief.confidence * 100, belief.status,
    )
    return (
        f"Belief '{hypothesis_id}' [{symbol}]: confidence={belief.confidence:.0%}, "
        f"+{belief.supports}/-{belief.contradicts}, status={belief.status}."
    )


@scientist_agent.tool
async def batch_judge(
    ctx: RunContext[AgentDeps],
    judgments: list[dict],
) -> str:
    """Judge multiple hypotheses at once against the same observation(s).

    Call this after recording an observation to update ALL relevant beliefs
    in one step. This is correct Bayesian updating: one piece of evidence
    updates every posterior it touches.

    Args:
        judgments: List of dicts, each with keys:
            - hypothesis_id: str
            - verdict: "supports" | "contradicts" | "neutral"
            - reasoning: str
            - observation_ids: list[str]
    """
    await ctx.deps.paused_event.wait()
    if ctx.deps.stop_event.is_set():
        raise RuntimeError("Agent stopped")

    results = []
    for j in judgments:
        hypothesis_id = j.get("hypothesis_id", "")
        verdict = j.get("verdict", "")
        reasoning = j.get("reasoning", "")
        observation_ids = j.get("observation_ids", [])

        belief = ctx.deps.world_state.get_belief(hypothesis_id)
        if not belief:
            results.append(f"  [{hypothesis_id}] NOT FOUND — skipped")
            continue

        if verdict not in ("supports", "contradicts", "neutral"):
            results.append(f"  [{hypothesis_id}] INVALID verdict '{verdict}' — skipped")
            continue

        if verdict == "supports":
            belief.supports += 1
        elif verdict == "contradicts":
            belief.contradicts += 1

        belief.last_updated = time.time()
        belief.update_status()
        ctx.deps.world_state.verdict_count += 1

        belief.history.append(BeliefSnapshot(
            iteration=ctx.deps.world_state.verdict_count,
            verdict=verdict,
            reasoning=reasoning,
            observation_ids=list(observation_ids),
        ))

        ctx.deps.world_state.upsert_belief(belief)

        symbol = {"supports": "+", "contradicts": "-", "neutral": "~"}[verdict]
        results.append(
            f"  [{hypothesis_id}] [{symbol}] confidence={belief.confidence:.0%}, "
            f"+{belief.supports}/-{belief.contradicts}, status={belief.status}"
        )

    ctx.deps.push_state_cb()
    ctx.deps.save_checkpoint_cb()

    return f"Batch judged {len(results)} beliefs:\n" + "\n".join(results)


@scientist_agent.tool
async def get_current_beliefs(ctx: RunContext[AgentDeps]) -> str:
    """Review all current hypotheses, their verdicts, and confidence scores.

    Call this to see what you have already proposed before deciding whether
    to add new hypotheses or judge existing ones with new evidence.
    """
    ws = ctx.deps.world_state

    lines = []

    if ws.entities:
        lines.append(f"Known entities: {', '.join(ws.entities)}")
        lines.append("")

    beliefs = ws.beliefs
    if not beliefs:
        lines.append("No hypotheses proposed yet.")
        return "\n".join(lines)

    lines.append(f"Current beliefs ({len(beliefs)} total, sorted by confidence):")
    for b in sorted(beliefs, key=lambda b: b.confidence, reverse=True):
        total_verdicts = b.supports + b.contradicts
        lines.append(
            f"  [{b.hypothesis.id}] {b.hypothesis.subject} {b.hypothesis.predicate} "
            f"{b.hypothesis.object_}: \"{b.hypothesis.description}\" "
            f"(confidence={b.confidence:.0%}, +{b.supports}/-{b.contradicts}, "
            f"status={b.status}, verdicts={total_verdicts})"
        )

    # -- Exploration guidance --
    lines.append("")
    lines.append("=== EXPLORATION GUIDANCE ===")

    # Most uncertain beliefs (near 50% confidence or low evidence)
    uncertain = sorted(
        beliefs,
        key=lambda b: abs(b.confidence - 0.5) + 0.1 * (b.supports + b.contradicts),
    )
    if uncertain:
        lines.append("Most uncertain (test these next):")
        for b in uncertain[:3]:
            lines.append(
                f"  -> [{b.hypothesis.id}] {b.hypothesis.subject} "
                f"{b.hypothesis.predicate} {b.hypothesis.object_} "
                f"(confidence={b.confidence:.0%}, verdicts={b.supports + b.contradicts})"
            )

    # Entity pairs with no hypothesis
    coverage = ws.pair_coverage()
    if coverage["uncovered"]:
        lines.append(f"Entity pairs with NO hypothesis ({len(coverage['uncovered'])}):")
        for subj, obj in coverage["uncovered"][:6]:
            lines.append(f"  -> {subj} <-> {obj}: no hypothesis yet — observe these together")

    lines.append("")
    lines.append(
        "STRATEGY: Design your next get_frames call to observe the most uncertain "
        "beliefs or uncovered pairs above. Focus on pairs you haven't tested."
    )

    return "\n".join(lines)
