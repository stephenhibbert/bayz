from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field


# -- Named observation bundles -------------------------------------------------


class Observation(BaseModel):
    """A named bundle of frames captured by the agent."""

    id: str = Field(description="Unique slug identifying this observation, e.g. 'obs-001'")
    description: str = Field(description="What was observed across these frames")
    frame_ids: list[int] = Field(default_factory=list, description="Frame IDs captured in this bundle")
    entities: list[str] = Field(default_factory=list, description="Entity names observed in these frames")
    timestamp: float = Field(default_factory=time.time)


# -- Hypotheses about interaction rules ----------------------------------------


class Hypothesis(BaseModel):
    """A proposed rule describing how one entity type interacts with another."""

    id: str = Field(description="Unique slug, e.g. 'pred-chases-wanderer'")
    subject: str = Field(description="The acting entity kind")
    predicate: str = Field(
        description="The interaction type: chases|flees|attracts|ignores|feeds_on"
    )
    object_: str = Field(description="The entity kind being acted upon", alias="object")
    description: str = Field(description="Plain English rule statement")

    model_config = {"populate_by_name": True}


# -- Verdict-based beliefs -----------------------------------------------------


class BeliefSnapshot(BaseModel):
    """A single verdict recorded against a belief."""

    iteration: int
    verdict: Literal["supports", "contradicts", "neutral"]
    reasoning: str = ""
    observation_ids: list[str] = Field(default_factory=list)


class Belief(BaseModel):
    """A hypothesis with confidence derived from verdict counts."""

    hypothesis: Hypothesis
    supports: int = 0
    contradicts: int = 0
    status: Literal["proposed", "strengthening", "confident", "refuted"] = "proposed"
    last_updated: float = Field(default_factory=time.time)
    history: list[BeliefSnapshot] = Field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Laplace-smoothed confidence from verdict counts."""
        return (self.supports + 1) / (self.supports + self.contradicts + 2)

    def update_status(self) -> None:
        c = self.confidence
        total = self.supports + self.contradicts
        if c >= 0.8 and total >= 3:
            self.status = "confident"
        elif c >= 0.65:
            self.status = "strengthening"
        elif c <= 0.25:
            self.status = "refuted"
        else:
            self.status = "proposed"


# -- World state ---------------------------------------------------------------


class WorldState(BaseModel):
    """The complete current state of the learning loop."""

    beliefs: list[Belief] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    loop_iteration: int = 0
    total_frames_observed: int = 0

    def get_belief(self, hypothesis_id: str) -> Belief | None:
        return next((b for b in self.beliefs if b.hypothesis.id == hypothesis_id), None)

    def upsert_belief(self, belief: Belief) -> None:
        existing = self.get_belief(belief.hypothesis.id)
        if existing:
            idx = self.beliefs.index(existing)
            self.beliefs[idx] = belief
        else:
            self.beliefs.append(belief)

    def get_observation(self, observation_id: str) -> Observation | None:
        return next((o for o in self.observations if o.id == observation_id), None)

    def add_observation(self, obs: Observation) -> None:
        if not self.get_observation(obs.id):
            self.observations.append(obs)

    def add_entity(self, name: str) -> None:
        if name not in self.entities:
            self.entities.append(name)

    def as_summary_dict(self) -> dict:
        """Serializable snapshot for WebSocket broadcast."""
        return {
            "loop_iteration": self.loop_iteration,
            "total_frames_observed": self.total_frames_observed,
            "entities": self.entities,
            "observations": {
                o.id: {
                    "id": o.id,
                    "description": o.description,
                    "frame_ids": o.frame_ids,
                    "entities": o.entities,
                }
                for o in self.observations
            },
            "beliefs": [
                {
                    "id": b.hypothesis.id,
                    "description": b.hypothesis.description,
                    "subject": b.hypothesis.subject,
                    "predicate": b.hypothesis.predicate,
                    "object": b.hypothesis.object_,
                    "supports": b.supports,
                    "contradicts": b.contradicts,
                    "confidence": round(b.confidence, 3),
                    "status": b.status,
                    "history": [
                        {
                            "iteration": s.iteration,
                            "verdict": s.verdict,
                            "reasoning": s.reasoning,
                            "observation_ids": s.observation_ids,
                        }
                        for s in b.history
                    ],
                }
                for b in sorted(self.beliefs, key=lambda b: b.confidence, reverse=True)
            ],
        }
