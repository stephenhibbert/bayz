"""Headless pygame-CE particle ecosystem simulator - config-driven."""

from __future__ import annotations

import io
import math
import os
import random

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

from scenarios import ScenarioConfig, InteractionRule, PREDATOR_PREY

WIDTH, HEIGHT = 160, 120
FPS = 30
ENTITY_RADIUS = 3
FEEDING_COLOR = (0, 220, 60)


class Entity:
    __slots__ = ("id", "kind", "x", "y", "vx", "vy", "state", "state_timer")

    def __init__(
        self,
        id: int,
        kind: str,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
    ):
        self.id = id
        self.kind = kind
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.state: str = "idle"
        self.state_timer: float = 0.0


class Simulator:
    """Config-driven pygame world. Tick-driven, headless-safe."""

    def __init__(self, config: ScenarioConfig | None = None, seed: int = 42):
        self.config = config or PREDATOR_PREY
        pygame.init()
        self.screen = pygame.Surface((WIDTH, HEIGHT))
        self.entities: list[Entity] = []
        self.tick_count: int = 0
        self._next_id = 0
        self._rng = random.Random(seed)
        self._last_frame: bytes = b""

        # Build color map from config
        self._colors: dict[str, tuple[int, int, int]] = {
            t.kind: t.color for t in self.config.entity_types
        }

        # Cache entity type properties
        self._type_props: dict[str, dict] = {
            t.kind: {
                "speed_cap": t.speed_cap,
                "random_walk": t.random_walk,
                "stationary": t.stationary,
                "special_render": t.special_render,
            }
            for t in self.config.entity_types
        }

        # Group rules by subject kind for efficient lookup
        self._rules_by_subject: dict[str, list[InteractionRule]] = {}
        for rule in self.config.rules:
            self._rules_by_subject.setdefault(rule.subject, []).append(rule)

        self._spawn_initial()
        self._render()
        self._snapshot_frame()

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _rand_pos(self, margin: int = 10) -> tuple[float, float]:
        return (
            self._rng.uniform(margin, WIDTH - margin),
            self._rng.uniform(margin, HEIGHT - margin),
        )

    def _rand_vel(self, mag: float = 20.0) -> tuple[float, float]:
        return (self._rng.uniform(-mag, mag), self._rng.uniform(-mag, mag))

    def _spawn_initial(self) -> None:
        # Count total stationary types to spread them apart
        stationary_types = [t for t in self.config.entity_types if t.stationary]
        stationary_idx = 0

        for typedef in self.config.entity_types:
            for i in range(typedef.count):
                if typedef.stationary:
                    # Spread all stationary entities in a ring
                    total_stationary = sum(t.count for t in stationary_types)
                    if total_stationary == 1:
                        x, y = WIDTH / 2, HEIGHT / 2
                    else:
                        angle = (stationary_idx / total_stationary) * math.pi * 2
                        r = min(WIDTH, HEIGHT) * 0.25
                        x = WIDTH / 2 + math.cos(angle) * r
                        y = HEIGHT / 2 + math.sin(angle) * r
                    stationary_idx += 1
                    self.entities.append(Entity(self._new_id(), typedef.kind, x, y))
                else:
                    x, y = self._rand_pos()
                    vx, vy = self._rand_vel(typedef.random_walk * 0.5)
                    self.entities.append(
                        Entity(self._new_id(), typedef.kind, x, y, vx, vy)
                    )

    # -- Physics ---------------------------------------------------------------

    def tick(self, dt: float = 1.0 / FPS) -> None:
        self._apply_rules(dt)
        self._apply_random_walk(dt)
        self._apply_physics(dt)
        self._clamp_speeds()
        self._bounce_walls()
        self._render()
        self._snapshot_frame()
        self.tick_count += 1

    def _apply_physics(self, dt: float) -> None:
        for e in self.entities:
            props = self._type_props.get(e.kind, {})
            if props.get("stationary"):
                continue
            e.x += e.vx * dt
            e.y += e.vy * dt
            e.vx *= 0.995
            e.vy *= 0.995

    def _apply_random_walk(self, dt: float) -> None:
        for e in self.entities:
            props = self._type_props.get(e.kind, {})
            walk = props.get("random_walk", 0)
            if walk > 0 and not props.get("stationary"):
                e.vx += self._rng.uniform(-walk, walk) * dt
                e.vy += self._rng.uniform(-walk, walk) * dt

    def _clamp_speeds(self) -> None:
        for e in self.entities:
            props = self._type_props.get(e.kind, {})
            cap = props.get("speed_cap", 40)
            speed = math.sqrt(e.vx * e.vx + e.vy * e.vy)
            if speed > cap:
                e.vx = e.vx / speed * cap
                e.vy = e.vy / speed * cap

    def _apply_rules(self, dt: float) -> None:
        # Decay state timers
        for e in self.entities:
            if e.state_timer > 0:
                e.state_timer -= dt
                if e.state_timer <= 0:
                    e.state = "idle"

        # Build entity-by-kind lookup
        by_kind: dict[str, list[Entity]] = {}
        for e in self.entities:
            by_kind.setdefault(e.kind, []).append(e)

        # Apply each rule
        for rule in self.config.rules:
            subjects = by_kind.get(rule.subject, [])
            objects_ = by_kind.get(rule.object_, [])
            if not subjects or not objects_:
                continue

            if rule.predicate == "chases":
                self._apply_chase(subjects, objects_, rule, dt)
            elif rule.predicate == "flees":
                self._apply_flee(subjects, objects_, rule, dt)
            elif rule.predicate == "attracts":
                self._apply_attract(subjects, objects_, rule, dt)
            # "ignores" is a no-op

    def _apply_chase(
        self,
        subjects: list[Entity],
        objects_: list[Entity],
        rule: InteractionRule,
        dt: float,
    ) -> None:
        for e in subjects:
            if e.state == "feeding":
                continue
            nearest = self._nearest(e, [o for o in objects_ if o is not e])
            if not nearest:
                continue
            dx = nearest.x - e.x
            dy = nearest.y - e.y
            dist = math.sqrt(dx * dx + dy * dy) or 0.01

            if dist < rule.contact_radius and rule.on_contact == "feed_respawn":
                e.state = "feeding"
                e.state_timer = 0.5
                nearest.x, nearest.y = self._rand_pos()
                nearest.vx, nearest.vy = self._rand_vel(20)
            elif dist < rule.radius:
                e.vx += (dx / dist) * rule.strength * dt
                e.vy += (dy / dist) * rule.strength * dt
                e.state = "chasing"

    def _apply_flee(
        self,
        subjects: list[Entity],
        objects_: list[Entity],
        rule: InteractionRule,
        dt: float,
    ) -> None:
        for e in subjects:
            nearest = self._nearest(e, [o for o in objects_ if o is not e])
            if not nearest:
                continue
            dist = self._dist(e, nearest)
            if dist < rule.radius:
                dx = e.x - nearest.x
                dy = e.y - nearest.y
                mag = max(math.sqrt(dx * dx + dy * dy), 0.01)
                scale = rule.strength * (1.0 - dist / rule.radius)
                e.vx += (dx / mag) * scale * dt
                e.vy += (dy / mag) * scale * dt
                e.state = "fleeing"
            elif e.state == "fleeing":
                e.state = "idle"

    def _apply_attract(
        self,
        subjects: list[Entity],
        objects_: list[Entity],
        rule: InteractionRule,
        dt: float,
    ) -> None:
        """Subject exerts gravitational pull on objects within radius."""
        for attractor in subjects:
            for obj in objects_:
                if obj is attractor:
                    continue
                dx = attractor.x - obj.x
                dy = attractor.y - obj.y
                dist_sq = dx * dx + dy * dy
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.01
                if dist > rule.radius:
                    continue
                force = rule.strength / (dist_sq + 100.0)
                obj.vx += (dx / dist) * force * dt
                obj.vy += (dy / dist) * force * dt

    def _bounce_walls(self) -> None:
        r = ENTITY_RADIUS
        for e in self.entities:
            props = self._type_props.get(e.kind, {})
            if props.get("stationary"):
                continue
            if e.x < r:
                e.x = r
                e.vx = abs(e.vx)
            elif e.x > WIDTH - r:
                e.x = WIDTH - r
                e.vx = -abs(e.vx)
            if e.y < r:
                e.y = r
                e.vy = abs(e.vy)
            elif e.y > HEIGHT - r:
                e.y = HEIGHT - r
                e.vy = -abs(e.vy)

    # -- Rendering -------------------------------------------------------------

    def _render(self) -> None:
        self.screen.fill((10, 10, 20))
        for e in self.entities:
            props = self._type_props.get(e.kind, {})
            if props.get("special_render") == "pulsing_ring":
                pulse = int(4 + 2 * abs(math.sin(self.tick_count * 0.08)))
                color = self._colors.get(e.kind, (255, 255, 255))
                dim = tuple(max(0, c - 40) for c in color)
                pygame.draw.circle(
                    self.screen, dim, (int(e.x), int(e.y)), pulse, 1
                )
                pygame.draw.circle(
                    self.screen, color, (int(e.x), int(e.y)), 2
                )
            else:
                if e.state == "feeding":
                    color = FEEDING_COLOR
                else:
                    color = self._colors.get(e.kind, (255, 255, 255))
                pygame.draw.circle(
                    self.screen, color, (int(e.x), int(e.y)), ENTITY_RADIUS
                )

    def _snapshot_frame(self) -> None:
        buf = io.BytesIO()
        pygame.image.save(self.screen, buf, "PNG")
        self._last_frame = buf.getvalue()

    def capture_frame(self) -> bytes:
        return self._last_frame

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _dist(a: Entity, b: Entity) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _nearest(source: Entity, candidates: list[Entity]) -> Entity | None:
        if not candidates:
            return None
        return min(candidates, key=lambda c: Simulator._dist(source, c))


# -- Standalone test -----------------------------------------------------------

if __name__ == "__main__":
    from scenarios import ALL_SCENARIOS

    for scenario in ALL_SCENARIOS:
        sim = Simulator(config=scenario)
        for _ in range(300):
            sim.tick()
        fname = f"frame_test_{scenario.id}.png"
        with open(fname, "wb") as f:
            f.write(sim.capture_frame())
        print(f"[{scenario.id}] Saved {fname} ({len(sim.capture_frame())} bytes)")
        print(f"  Entities: {[(e.kind, round(e.x), round(e.y)) for e in sim.entities]}")
