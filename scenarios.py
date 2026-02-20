"""Scenario configurations and ground-truth assertions for the Bayz system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntityTypeDef:
    kind: str
    color: tuple[int, int, int]
    count: int
    speed_cap: float = 40.0
    random_walk: float = 20.0  # noise magnitude (0 = stationary)
    stationary: bool = False
    special_render: str | None = None  # "pulsing_ring"


@dataclass(frozen=True)
class InteractionRule:
    subject: str  # entity kind performing the action
    predicate: str  # chases | flees | attracts | ignores
    object_: str  # entity kind being acted upon
    strength: float = 60.0  # force magnitude
    radius: float = 999.0  # effective range (999 = unlimited)
    on_contact: str | None = None  # "feed_respawn" | None
    contact_radius: float = 6.0


@dataclass(frozen=True)
class ScenarioConfig:
    id: str
    name: str
    description: str
    entity_types: list[EntityTypeDef]
    rules: list[InteractionRule]
    assertions: list[str]  # Binary ground-truth assertions for evaluation



# ---------------------------------------------------------------------------
# Scenario 1: Predator-Prey Ecosystem
# ---------------------------------------------------------------------------

PREDATOR_PREY = ScenarioConfig(
    id="predator-prey",
    name="Predator-Prey Ecosystem",
    description=(
        "A classic ecosystem with predators chasing prey, an attractor pulling "
        "everything inward, and inert debris drifting around."
    ),
    entity_types=[
        EntityTypeDef("wanderer", (0, 200, 200), count=6, speed_cap=40, random_walk=25),
        EntityTypeDef("predator", (220, 40, 40), count=2, speed_cap=50, random_walk=5),
        EntityTypeDef(
            "attractor", (220, 200, 0), count=1, stationary=True,
            random_walk=0, special_render="pulsing_ring",
        ),
        EntityTypeDef("debris", (120, 120, 120), count=4, speed_cap=20, random_walk=8),
    ],
    rules=[
        InteractionRule("predator", "chases", "wanderer", strength=55, on_contact="feed_respawn"),
        InteractionRule("wanderer", "flees", "predator", strength=80, radius=45),
        InteractionRule("attractor", "attracts", "wanderer", strength=600),
        InteractionRule("attractor", "attracts", "predator", strength=600),
        InteractionRule("attractor", "attracts", "debris", strength=600),
    ],
    assertions=[
        "Red entities actively pursue or chase cyan entities",
        "Cyan entities flee or run away from red entities",
        "The stationary yellow entity pulls or attracts other entities toward it",
        "Grey entities are inert and do not meaningfully chase, flee, or interact",
        "Red entities briefly flash green when they catch a cyan entity (feeding behavior)",
    ],
)

# ---------------------------------------------------------------------------
# Scenario 2: Tribal Territories
# ---------------------------------------------------------------------------

TRIBAL_TERRITORIES = ScenarioConfig(
    id="tribal-territories",
    name="Tribal Territories",
    description=(
        "Two rival tribes compete for territory. Each tribe flocks together and "
        "has a stationary beacon. Neutral observers wander freely."
    ),
    entity_types=[
        EntityTypeDef("tribe-a", (40, 80, 220), count=4, speed_cap=40, random_walk=15),
        EntityTypeDef("tribe-b", (220, 130, 30), count=4, speed_cap=45, random_walk=15),
        EntityTypeDef(
            "beacon-a", (180, 180, 255), count=1, stationary=True,
            random_walk=0, special_render="pulsing_ring",
        ),
        EntityTypeDef(
            "beacon-b", (255, 180, 100), count=1, stationary=True,
            random_walk=0, special_render="pulsing_ring",
        ),
        EntityTypeDef("neutral", (40, 200, 80), count=3, speed_cap=25, random_walk=20),
    ],
    rules=[
        InteractionRule("tribe-b", "chases", "tribe-a", strength=50, on_contact="feed_respawn"),
        InteractionRule("tribe-a", "flees", "tribe-b", strength=70, radius=50),
        InteractionRule("beacon-a", "attracts", "tribe-a", strength=400),
        InteractionRule("beacon-b", "attracts", "tribe-b", strength=400),
        InteractionRule("tribe-a", "attracts", "tribe-a", strength=200, radius=60),
        InteractionRule("tribe-b", "attracts", "tribe-b", strength=200, radius=60),
    ],
    assertions=[
        "Orange entities chase or pursue blue entities",
        "Blue entities flee or run away from orange entities",
        "The light-blue stationary beacon attracts blue entities toward it",
        "The light-orange stationary beacon attracts orange entities toward it",
        "Green entities are neutral and are not chased, do not flee, and do not meaningfully interact",
        "Same-colored entities flock or cluster together (intra-tribe attraction)",
    ],
)

# ---------------------------------------------------------------------------
# Scenario 3: Magnetic Fields
# ---------------------------------------------------------------------------

MAGNETIC_FIELDS = ScenarioConfig(
    id="magnetic-fields",
    name="Magnetic Fields",
    description=(
        "Charged particles interact: like charges repel, opposites attract. "
        "Conductors are drawn to everything charged. Insulators drift inertly."
    ),
    entity_types=[
        EntityTypeDef("positive", (220, 50, 50), count=4, speed_cap=35, random_walk=12),
        EntityTypeDef("negative", (50, 80, 220), count=4, speed_cap=35, random_walk=12),
        EntityTypeDef("conductor", (220, 200, 50), count=3, speed_cap=30, random_walk=10),
        EntityTypeDef("insulator", (120, 120, 120), count=3, speed_cap=15, random_walk=8),
    ],
    rules=[
        InteractionRule("positive", "flees", "positive", strength=70, radius=40),
        InteractionRule("negative", "flees", "negative", strength=70, radius=40),
        InteractionRule("positive", "attracts", "negative", strength=400, radius=80),
        InteractionRule("negative", "attracts", "positive", strength=400, radius=80),
        InteractionRule("positive", "attracts", "conductor", strength=300, radius=70),
        InteractionRule("negative", "attracts", "conductor", strength=300, radius=70),
    ],
    assertions=[
        "Red entities repel or flee from other red entities (same-type repulsion)",
        "Blue entities repel or flee from other blue entities (same-type repulsion)",
        "Red and blue entities attract each other (opposite-type attraction)",
        "Yellow entities are attracted to both red and blue entities",
        "Grey entities are inert and do not meaningfully interact with anything",
    ],
)

# ---------------------------------------------------------------------------

ALL_SCENARIOS: list[ScenarioConfig] = [PREDATOR_PREY, TRIBAL_TERRITORIES, MAGNETIC_FIELDS]

SCENARIOS_BY_ID: dict[str, ScenarioConfig] = {s.id: s for s in ALL_SCENARIOS}


def get_scenario(scenario_id: str) -> ScenarioConfig:
    if scenario_id not in SCENARIOS_BY_ID:
        raise ValueError(f"Unknown scenario: {scenario_id}. Available: {list(SCENARIOS_BY_ID)}")
    return SCENARIOS_BY_ID[scenario_id]
