from pydantic import BaseModel, field_validator, ValidationInfo


class Hyperparameter(BaseModel):
    alpha: float
    "Learning rate, determines the extent to which our Q-values are being updated in every iteration"
    gamma: float
    "Discount factor, determines the importance of future rewards"
    epsilon: float
    "Exploration rate, determines the probability of choosing a random action instead of one based on already learned Q-values"
    min_epsilon: float
    epsilon_decay_rate: float
    episodes_training: int
    episodes_testing: int
    
    @field_validator("alpha", "gamma", "epsilon", "epsilon_decay_rate")
    @classmethod
    def constrained_between_zero_and_one(cls, value: float, info: ValidationInfo) -> float:
        if not 0 < value < 1:
            raise ValueError(f"Hyperparameter {info.field_name} must be such that 0 < {info.field_name} < 1. Got {value} instead.")
        return value
    
    @field_validator("episodes_training", "episodes_testing")
    @classmethod
    def constrained_positive(cls, value: int, info: ValidationInfo) -> int:
        if value <= 0:
            raise ValueError(f"Hyperparameter {info.field_name} must be a positive integer. Got {value} instead.")
        return value
    
    @field_validator("min_epsilon")
    @classmethod
    def constrained_less_than_epsilon(cls, value: float, info: ValidationInfo) -> float:
        if value >= info.data["epsilon"]:
            raise ValueError(f"Hyperparameter min_epsilon must be less than epsilon. Got {value} >= {info.data['epsilon']} instead.")
        return value
