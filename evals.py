"""Pydantic Evals harness for evaluating the Bayz learning loop."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge, OutputConfig

from learning_loop import LearningLoop
from models import WorldState
from scenarios import ALL_SCENARIOS, ScenarioConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

MODEL = "anthropic:claude-sonnet-4-5-20250929"

# How many belief-update iterations to wait for before evaluating
DEFAULT_TARGET_ITERATIONS = 3


def format_beliefs_for_eval(world_state: WorldState) -> str:
    """Format the world state beliefs as human-readable text for LLMJudge."""
    if not world_state.beliefs:
        return "No beliefs were formed. The system failed to generate any hypotheses."

    lines = [
        f"After {world_state.loop_iteration} learning cycles "
        f"({world_state.total_frames_observed} frames observed), "
        f"the system holds {len(world_state.beliefs)} beliefs:\n"
    ]
    for b in sorted(world_state.beliefs, key=lambda b: b.posterior, reverse=True):
        lines.append(
            f"- [{b.status.upper()}] {b.hypothesis.description} "
            f"({b.hypothesis.subject} {b.hypothesis.predicate} {b.hypothesis.object_}) "
            f"| confidence: {b.posterior:.0%} | observations: {b.observation_count}"
        )
    return "\n".join(lines)


async def evaluate_assertions(
    beliefs_text: str,
    assertions: list[str],
    scenario_id: str = "scenario",
) -> list[dict]:
    """Evaluate assertions via Dataset.evaluate() so results appear in logfire evals dashboard."""

    def _rubric(assertion: str) -> str:
        return (
            f"Does the agent's belief system correctly capture the following?\n\n"
            f'"{assertion}"\n\n'
            f"Answer YES if the beliefs clearly support this with reasonable confidence "
            f"(posterior > 0.5). Answer NO if the assertion is missing, contradicted, "
            f"or only weakly supported."
        )

    cases = [
        Case(
            name=f"{scenario_id} / {i + 1}",
            inputs=beliefs_text,
            expected_output=None,
            evaluators=(
                LLMJudge(
                    rubric=_rubric(assertion),
                    model=MODEL,
                    include_input=False,
                    score=OutputConfig(include_reason=False),
                    assertion=OutputConfig(include_reason=True),
                ),
            ),
            metadata={"assertion": assertion},
        )
        for i, assertion in enumerate(assertions)
    ]

    ds = Dataset(name=f"bayz / {scenario_id}", cases=cases)

    async def passthrough(beliefs: str) -> str:
        return beliefs

    report = await ds.evaluate(passthrough, progress=False)

    results = []
    for case_result, assertion in zip(report.cases, assertions):
        pass_result = case_result.assertions.get("LLMJudge_pass")
        score_result = case_result.scores.get("LLMJudge_score")
        passed = bool(pass_result.value) if pass_result else False
        score = float(score_result.value) if score_result else (1.0 if passed else 0.0)
        reason = (pass_result.reason or "") if pass_result else ""
        results.append({
            "assertion": assertion,
            "passed": passed,
            "score": score,
            "reason": reason,
        })
        log.info(
            "Assertion [%s]: %s — %s",
            "PASS" if passed else "FAIL",
            assertion[:60],
            reason[:80],
        )
    return results


async def run_scenario_task(scenario: ScenarioConfig) -> str:
    """Run the learning loop for N iterations and return belief summary."""
    target = DEFAULT_TARGET_ITERATIONS
    log.info(
        "Starting eval for scenario '%s' (target: %d iterations)",
        scenario.id,
        target,
    )

    loop = LearningLoop(scenario)
    loop.start()

    try:
        timeout = 300  # 5 minute max per scenario
        start = asyncio.get_event_loop().time()
        while loop.world_state.loop_iteration < target:
            if asyncio.get_event_loop().time() - start > timeout:
                log.warning("Timeout reached for scenario %s", scenario.id)
                break
            await asyncio.sleep(1)
    finally:
        loop.stop()

    result = format_beliefs_for_eval(loop.world_state)
    log.info("Eval for '%s' complete:\n%s", scenario.id, result)
    return result


# -- Build the pydantic-evals dataset -----------------------------------------
# For the CLI harness we combine all assertions into a single rubric string.

def _assertions_to_rubric(assertions: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(assertions))
    return (
        f"Evaluate whether the agent's beliefs capture the following rules. "
        f"Score highly only if most are correctly identified:\n\n{numbered}"
    )


dataset = Dataset(
    name="bayz-scenario-evaluation",
    cases=[
        Case(
            name=scenario.id,
            inputs=scenario.id,
            expected_output=None,
            metadata={
                "description": scenario.description,
                "assertions": scenario.assertions,
            },
            evaluators=(
                LLMJudge(
                    rubric=_assertions_to_rubric(scenario.assertions),
                    include_input=True,
                ),
            ),
        )
        for scenario in ALL_SCENARIOS
    ],
)


async def eval_task(scenario_id: str) -> str:
    """The task function that pydantic-evals calls for each case."""
    from scenarios import get_scenario
    scenario = get_scenario(scenario_id)
    return await run_scenario_task(scenario)


async def run_evals():
    """Run the full evaluation suite."""
    report = await dataset.evaluate(eval_task)
    report.print(include_input=True, include_output=True)


if __name__ == "__main__":
    import logfire
    from dotenv import load_dotenv

    load_dotenv()
    logfire.configure(
        service_name="bayz-evals",
        environment=os.environ.get("RAILWAY_ENVIRONMENT", "development"),
    )
    logfire.instrument_pydantic_ai()

    if len(sys.argv) > 1:
        DEFAULT_TARGET_ITERATIONS = int(sys.argv[1])
    asyncio.run(run_evals())
