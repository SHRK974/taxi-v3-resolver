from pydantic import BaseModel, ConfigDict
from typing import Any, Self

class StepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    state: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]