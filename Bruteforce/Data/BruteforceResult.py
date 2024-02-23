from pydantic import BaseModel

from Bruteforce.Data.EpisodeResult import EpisodeResult


class BruteforceResult(BaseModel):
    total_solved: int
    total_unsolved: int
    total_attempts: int
    success_rate: float
    results: list[EpisodeResult]