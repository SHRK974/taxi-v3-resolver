from pydantic import BaseModel


class TuningQuantifier(BaseModel):
    success_rate: float
    success_rate_weight: int = 2
    "The higher the weight, the more success rate is rewarding the total score."
    mean_rewards_per_steps: float
    mean_rewards_per_steps_weight: int = 1.5
    "The higher the weight, the more rewards are rewarding the total score."
    mean_penalties_per_episode: float
    mean_penalties_per_episode_weight: int = 3
    "The higher the weight, the more penalties are penalizing the total score."
    mean_steps_per_episode: float
    mean_steps_per_episode_weight: int = 1
    "The higher the weight, the more steps are rewarding the total score."
    
    def calculate_score(self) -> float:
        """
        Calculate the score of the tuning quantifier
        """
        adjusted_success_rate = self.success_rate * self.success_rate_weight
        adjusted_mean_rewards_per_steps = self.mean_rewards_per_steps * self.mean_rewards_per_steps_weight
        adjusted_mean_penalties_per_episode = self.mean_penalties_per_episode * self.mean_penalties_per_episode_weight
        adjusted_mean_steps_per_episode = self.mean_steps_per_episode * self.mean_steps_per_episode_weight
        return (
            adjusted_success_rate
            + adjusted_mean_rewards_per_steps
            - adjusted_mean_steps_per_episode
            - adjusted_mean_penalties_per_episode
        )
