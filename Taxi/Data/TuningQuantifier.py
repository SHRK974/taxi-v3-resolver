from pydantic import BaseModel

from Taxi.Data.EpisodeResult import EpisodeResult


class TuningQuantifier(BaseModel):
    success_rate: float
    success_rate_weight: int = 2
    "The higher the weight, the more success rate is rewarding the total score."
    mean_rewards_per_steps: float
    mean_rewards_per_steps_weight: int = 1.5
    "The higher the weight, the more rewards are rewarding the total score."
    mean_penalties_per_episode: float
    mean_penalties_per_episode_weight: int = 100
    "The higher the weight, the more penalties are penalizing the total score."
    best_episode: EpisodeResult
    worst_episode: EpisodeResult
    
    def calculate_score(self) -> float:
        """
        Calculate the score of the tuning quantifier
        """
        success_rate_penalty_factor = 0.5 if self.success_rate < 100 else 1
        worst_episode_penalty = self.worst_episode.epochs / self.best_episode.epochs
        
        adjusted_success_rate = self.success_rate * self.success_rate_weight
        adjusted_mean_rewards_per_steps = self.mean_rewards_per_steps * self.mean_rewards_per_steps_weight
        adjusted_mean_penalties_per_episode = self.mean_penalties_per_episode * self.mean_penalties_per_episode_weight
        
        return (
            adjusted_success_rate
            + adjusted_mean_rewards_per_steps
            - adjusted_mean_penalties_per_episode
        ) * success_rate_penalty_factor - worst_episode_penalty
