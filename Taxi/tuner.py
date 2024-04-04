import sys
from os.path import abspath, dirname, join

import optuna
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from Q_Learning.Models.HyperparameterTuning import objective as q_learning_objective
from Sarsa.Models.HyperparameterTuning import objective as sarsa_objective

WORKERS = 10 # Might need to adjust this number based on the available resources on your machine

if __name__ == "__main__":
    study_q_learning = optuna.create_study(
        storage="sqlite:///tuning.sqlite3",
        study_name="tuning_q_learning",
        load_if_exists=True,
        direction="maximize",
    )
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for _ in range(10):
            pool.submit(study_q_learning.optimize, q_learning_objective, n_trials=10)
    
    study_sarsa = optuna.create_study(
        storage="sqlite:///tuning.sqlite3",
        study_name="tuning_sarsa",
        load_if_exists=True,
        direction="maximize",
    )
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for _ in range(10):
            pool.submit(study_sarsa.optimize, sarsa_objective, n_trials=10)