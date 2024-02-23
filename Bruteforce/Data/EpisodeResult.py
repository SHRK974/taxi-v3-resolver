from pydantic import BaseModel


class EpisodeResult(BaseModel):
    solved: bool
    total_reward: float
    total_steps: int