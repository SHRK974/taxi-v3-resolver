from typing import Any, SupportsFloat

from pydantic import BaseModel


class StepResult(BaseModel):
    state: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

def step_result_from_tuple(step: tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]) -> StepResult:
    """
    Creates a StepResult from a tuple.

    Args:
        step (tuple): The step from the environment.

    Returns:
        StepResult: The step result.
    """
    state, reward, terminated, truncated, info = step

    return StepResult(state=state, reward=reward, terminated=terminated, truncated=truncated, info=info)
