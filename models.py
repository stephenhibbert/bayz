from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# -- Named observation bundles -------------------------------------------------


class Observation(BaseModel):
    """A named bundle of frames captured by the agent."""

    id: str = Field(description="Unique slug identifying this observation, e.g. 'obs-001'")
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
    observation_count: int
    status: str
    reasoning: str = ""
    observation_ids: list[str] = Field(default_factory=list, description="Observations used in this update")

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, data: dict) -> dict:
        if isinstance(data, dict):
            if "evidence_count" in data and "observation_count" not in data:
                data["observation_count"] = data.pop("evidence_count")
            if "evidence_ids" in data and "observation_ids" not in data:
                data["observation_ids"] = data.pop("evidence_ids")
        return data


class Belief(BaseModel):
    """A hypothesis with a Bayesian-style confidence score."""

    hypothesis: Hypothesis
    prior: float = Field(default=0.5, ge=0.0, le=1.0)
    likelihood: float = Field(
        default=0.5, ge=0.0, le=1.0, description="P(observations | hypothesis true)"
    )
    posterior: float = Field(default=0.5, ge=0.0, le=1.0, description="Updated belief score")
    observation_count: int = 0
    observation_ids: list[str] = Field(
        default_factory=list,
        description="IDs of Observation bundles that have contributed to this belief",
    )
    status: Literal["proposed", "strengthening", "confident", "refuted"] = "proposed"
    last_updated: float = Field(default_factory=time.time)
    history: list[BeliefSnapshot] = Field(default_factory=list)
    origin_reasoning: str = Field(default="", description="Why the hypothesis agent proposed this")

    def update_status(self) -> None:
        if self.posterior >= 0.85 and self.observation_count >= 3:
            self.status = "confident"
        elif self.posterior >= 0.65:
            self.status = "strengthening"
        elif self.posterior < 0.2:
            self.status = "refuted"
        else:
            self.status = "proposed"

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, data: dict) -> dict:
        if isinstance(data, dict):
            if "evidence_count" in data and "observation_count" not in data:
                data["observation_count"] = data.pop("evidence_count")
            if "evidence_ids" in data and "observation_ids" not in data:
                data["observation_ids"] = data.pop("evidence_ids")
        return data


# -- World state ---------------------------------------------------------------


class WorldState(BaseModel):
    """The complete current state of the Bayesian learning loop."""

    beliefs: list[Belief] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
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

    def get_observation(self, observation_id: str) -> Observation | None:
        return next((o for o in self.observations if o.id == observation_id), None)

    def add_observation(self, obs: Observation) -> None:
        if not self.get_observation(obs.id):
            self.observations.append(obs)

    def as_summary_dict(self) -> dict:
        """Serializable snapshot for WebSocket broadcast."""
        return {
            "loop_iteration": self.loop_iteration,
            "total_frames_observed": self.total_frames_observed,
            "observations": {
                o.id: {
                    "id": o.id,
                    "description": o.description,
                    "frame_ids": o.frame_ids,
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
                    "prior": round(b.prior, 3),
                    "likelihood": round(b.likelihood, 3),
                    "posterior": round(b.posterior, 3),
                    "observation_count": b.observation_count,
                    "observation_ids": b.observation_ids,
                    "status": b.status,
                    "origin_reasoning": b.origin_reasoning,
                    "history": [
                        {
                            "iteration": s.iteration,
                            "prior": round(s.prior, 3),
                            "likelihood": round(s.likelihood, 3),
                            "posterior": round(s.posterior, 3),
                            "observation_count": s.observation_count,
                            "status": s.status,
                            "reasoning": s.reasoning,
                            "observation_ids": s.observation_ids,
                        }
                        for s in b.history
                    ],
                }
                for b in sorted(self.beliefs, key=lambda b: b.posterior, reverse=True)
            ],
        }

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, data: dict) -> dict:
        if isinstance(data, dict):
            if "evidence" in data and "observations" not in data:
                data["observations"] = data.pop("evidence")
            data.pop("recent_observations", None)
        return data


# -- Agent input/output contracts ---------------------------------------------


class HypothesisAgentOutput(BaseModel):
    new_hypotheses: list[Hypothesis] = Field(default_factory=list)
    reasoning: str = ""


class BeliefUpdaterOutput(BaseModel):
    updated_beliefs: list[Belief] = Field(default_factory=list)
    reasoning: str = ""
