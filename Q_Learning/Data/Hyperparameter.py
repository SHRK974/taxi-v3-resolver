from pydantic import BaseModel

class Hyperparameter(BaseModel):
    alpha: float # Learning rate, determines to what extent the newly acquired information will override the old information
    gamma: float # Discount factor, determines the importance of future rewards
    epsilon: float # Exploration vs exploitation trade-off, higher value means more exploration
    episodes_training: int
    episodes_testing: int