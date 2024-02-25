import numpy as np
from pydantic import BaseModel

from Taxi.Data.EpisodeResult import EpisodeResult


class BatchResult(BaseModel):
    total_solved: int
    total_unsolved: int
    total_attempts: int
    success_rate: float
    results: list[EpisodeResult]
    
    def summary(self) -> None:
        """
        Print the batch results
        """
        print(f"Solved: {self.total_solved}, Unsolved: {self.total_unsolved}, Total: {self.total_attempts} ({self.success_rate}% success rate)\n")
        results = self.results
        
        rewards = [result.rewards for result in results]
        penalties = [result.penalties for result in results]
        epochs = [result.epochs for result in results]
        
        print(f"Mean rewards per steps: {np.mean(rewards) / np.mean(epochs)}")
        print(f"Mean penalties per episode: {np.mean(penalties)}")
        print(f"Mean steps per episode: {np.mean(epochs)}\n")
        
        best_result = next(result for result in results if result.rewards == np.max(rewards))
        print(f"Best episode: {best_result.rewards} rewards, {best_result.penalties} penalties, {best_result.epochs} steps")
        worst_result = next(result for result in results if result.rewards == np.min(rewards))
        print(f"Worst episode: {worst_result.rewards} rewards, {worst_result.penalties} penalties, {worst_result.epochs} steps")