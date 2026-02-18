from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field


# -- What the Observer sees in a single frame ----------------------------------


class ObservedEntity(BaseModel):
    """A single entity identified by the vision agent in one frame."""

    label: str = Field(description="Short descriptive label, e.g. 'cyan dot', 'red fast mover'")
    kind_guess: str = Field(
        description="Your inferred label for this entity type based solely on what you observe"
    )
    x_pct: float = Field(ge=0.0, le=1.0, description="Horizontal position as fraction of frame width")
    y_pct: float = Field(ge=0.0, le=1.0, description="Vertical position as fraction of frame height")
    motion: Literal["stationary", "slow", "fast", "erratic"] = "slow"
    notes: str = Field(default="", description="Any observed behavior or anomaly")


class FrameObservation(BaseModel):
    """All entities seen in one captured frame."""

    frame_id: int = 0
    timestamp: float = Field(default_factory=time.time)
    entities: list[ObservedEntity] = Field(default_factory=list)
    scene_summary: str = Field(
        default="", description="One-sentence description of the overall scene state"
    )


# -- Named evidence bundles ----------------------------------------------------


class Evidence(BaseModel):
    """A named, reusable bundle of frames collected as evidence for/against hypotheses."""

    id: str = Field(description="Unique slug identifying this evidence, e.g. 'ev-001'")
    description: str = Field(description="What was observed across these frames")
    frame_ids: list[int] = Field(default_factory=list, description="Frame IDs captured in this bundle")
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


# -- Scored beliefs ------------------------------------------------------------


class BeliefSnapshot(BaseModel):
    """A point-in-time record of a belief's Bayesian quantities at one iteration."""

    iteration: int
    prior: float
    likelihood: float
    posterior: float
    evidence_count: int
    status: str
    reasoning: str = ""
    evidence_ids: list[str] = Field(default_factory=list, description="Evidence used in this update")


class Belief(BaseModel):
    """A hypothesis with a Bayesian-style confidence score."""

    hypothesis: Hypothesis
    prior: float = Field(default=0.5, ge=0.0, le=1.0)
    likelihood: float = Field(
        default=0.5, ge=0.0, le=1.0, description="P(evidence | hypothesis true)"
    )
    posterior: float = Field(default=0.5, ge=0.0, le=1.0, description="Updated belief score")
    evidence_count: int = 0
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="IDs of Evidence bundles that have contributed to this belief",
    )
    status: Literal["proposed", "strengthening", "confident", "refuted"] = "proposed"
    last_updated: float = Field(default_factory=time.time)
    history: list[BeliefSnapshot] = Field(default_factory=list)
    origin_reasoning: str = Field(default="", description="Why the hypothesis agent proposed this")

    def update_status(self) -> None:
        if self.posterior >= 0.85 and self.evidence_count >= 3:
            self.status = "confident"
        elif self.posterior >= 0.65:
            self.status = "strengthening"
        elif self.posterior < 0.2:
            self.status = "refuted"
        else:
            self.status = "proposed"


# -- World state ---------------------------------------------------------------


class WorldState(BaseModel):
    """The complete current state of the Bayesian learning loop."""

    beliefs: list[Belief] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    recent_observations: list[FrameObservation] = Field(default_factory=list)
    loop_iteration: int = 0
    total_frames_observed: int = 0
    last_hypothesis_run: float = 0.0
    last_belief_update: float = 0.0

    def get_belief(self, hypothesis_id: str) -> Belief | None:
        return next((b for b in self.beliefs if b.hypothesis.id == hypothesis_id), None)

    def upsert_belief(self, belief: Belief) -> None:
        existing = self.get_belief(belief.hypothesis.id)
        if existing:
            idx = self.beliefs.index(existing)
            self.beliefs[idx] = belief
        else:
            self.beliefs.append(belief)

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        return next((e for e in self.evidence if e.id == evidence_id), None)

    def add_evidence(self, ev: Evidence) -> None:
        if not self.get_evidence(ev.id):
            self.evidence.append(ev)

    def as_summary_dict(self) -> dict:
        """Serializable snapshot for WebSocket broadcast."""
        return {
            "loop_iteration": self.loop_iteration,
            "total_frames_observed": self.total_frames_observed,
            "evidence": {
                e.id: {
                    "id": e.id,
                    "description": e.description,
                    "frame_ids": e.frame_ids,
                }
                for e in self.evidence
            },
            "beliefs": [
                {
                    "id": b.hypothesis.id,
                    "description": b.hypothesis.description,
                    "subject": b.hypothesis.subject,
                    "predicate": b.hypothesis.predicate,
                    "object": b.hypothesis.object_,
                    "prior": round(b.prior, 3),
                    "likelihood": round(b.likelihood, 3),
                    "posterior": round(b.posterior, 3),
                    "evidence_count": b.evidence_count,
                    "evidence_ids": b.evidence_ids,
                    "status": b.status,
                    "origin_reasoning": b.origin_reasoning,
                    "history": [
                        {
                            "iteration": s.iteration,
                            "prior": round(s.prior, 3),
                            "likelihood": round(s.likelihood, 3),
                            "posterior": round(s.posterior, 3),
                            "evidence_count": s.evidence_count,
                            "status": s.status,
                            "reasoning": s.reasoning,
                            "evidence_ids": s.evidence_ids,
                        }
                        for s in b.history
                    ],
                }
                for b in sorted(self.beliefs, key=lambda b: b.posterior, reverse=True)
            ],
            "recent_scene": (
                self.recent_observations[-1].scene_summary
                if self.recent_observations
                else ""
            ),
        }


# -- Agent input/output contracts ---------------------------------------------


class HypothesisAgentOutput(BaseModel):
    new_hypotheses: list[Hypothesis] = Field(default_factory=list)
    reasoning: str = ""


class BeliefUpdaterOutput(BaseModel):
    updated_beliefs: list[Belief] = Field(default_factory=list)
    reasoning: str = ""
