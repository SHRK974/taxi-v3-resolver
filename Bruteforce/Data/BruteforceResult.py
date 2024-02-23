from pydantic import BaseModel

class BruteforceResult(BaseModel):
    solved: bool
    total_reward: float
    total_steps: int