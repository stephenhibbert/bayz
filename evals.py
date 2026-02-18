"""Pydantic Evals harness for evaluating the Bayz learning loop."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge
from pydantic_evals.evaluators.llm_as_a_judge import judge_output

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
            f"| confidence: {b.posterior:.0%} | evidence: {b.evidence_count} observations"
        )
    return "\n".join(lines)


async def evaluate_assertions(
    beliefs_text: str,
    assertions: list[str],
) -> list[dict]:
    """Run each assertion as an independent binary judge call."""
    results = []
    for assertion in assertions:
        rubric = (
            f"Does the agent's belief system correctly capture the following?\n\n"
            f'"{assertion}"\n\n'
            f"Answer YES if the beliefs clearly support this with reasonable confidence "
            f"(posterior > 0.5). Answer NO if the assertion is missing, contradicted, "
            f"or only weakly supported."
        )
        grading = await judge_output(output=beliefs_text, rubric=rubric, model=MODEL)
        results.append({
            "assertion": assertion,
            "passed": grading.pass_,
            "score": grading.score,
            "reason": grading.reason,
        })
        log.info(
            "Assertion [%s]: %s — %s",
            "PASS" if grading.pass_ else "FAIL",
            assertion[:60],
            grading.reason[:80],
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
