from pydantic import BaseModel


class Hyperparameter(BaseModel):
    alpha: float  # Learning rate
    gamma: float  # Discount factor, determines the importance of future rewards
    epsilon: float  # Exploration vs exploitation trade-off, higher value means more exploration
    episodes_training: int
    episodes_testing: int
