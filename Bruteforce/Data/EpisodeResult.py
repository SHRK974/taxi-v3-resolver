from pydantic import BaseModel


class EpisodeResult(BaseModel):
    solved: bool
    epochs: int
    rewards: float
    penalties: int