# T-AIA-902 - Taxi Driver

## Project description

The goal of this project is to solve the Taxi-v3 game with an optimized model-free episodic learning algorithm. The agent is a taxi that must pick up some random passengers, and drop them off at some specific locations. The game environment can be found in the Gym library.

> You are free to choose the algorithm, whether on-policy or off-policy.
> We recommend Deep Q-Learning, or Monte Carlo-based algorithms, but feel free to find the best possible approach.

### Expected features

Your program must include at least 2 modes :

- user mode allows to tune algorithm parameters
- time-limited mode uses optimized parameters to reduce the steps to solve the problem in a given time.

> Both training and testing the number of episodes must be entered by the user when the program launches.

### Expected output

Your program outputs (at least) the `mean time for finishing the game` and `mean rewards within all the episodes`. It should also display random episodes.

To **prove** your algo performances, you must benchmark it against as many metrics as possible, playing around with several parameters and rewards definition.

You are also expected to develop a naive bruteforce algorithm as a comparaison point.

### Delivery

You must report these facts and figures in a document that also includes :

- commentaries about this benchmark that justify your algorithm choices and tuning
- your optimization strategy (concerning parameters and game rewards)
- different relevant graphics and arrays

> To prove the efficiency of your algorithm, you may add other comparaison algorithms.
> For instance, a Q-Learning algo might be compared to a Deep Q-Learning algo.

## Usage

### Installation

```bash
pip install -r requirements.txt
```

## Bruteforce

A naive bruteforce algorithm is available to compare with your optimized algorithm. To run it, use the following command:

```bash
python Bruteforce/main.py --episodes 1000
```

> The `--episodes` argument is optional and defaults to 10000. You are free to change it to any value you want.

### Strategy

The current bruteforce implementation is not naive, in the sens that it will not try every random action each episode until the game is solved as it would be time-consuming. The goal was to find a solution to start the game with a predictable and repeatable behavior.

The strategy was designed around the grid layout of the game *(see figure below)*. There are 5 lanes the taxi can move on, and walls preventing the taxi passing through them. The particularity of this layout is, wherever the taxi is, if it goes up, it will never be blocked by a wall.

```bash
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
```

The sequence of actions is as follows:

- Move the taxi up, until it hits the wall.
- Move the taxi left, until it hits the wall. At this point, the taxi is in the first or the third column.
- Move the taxi right, until it hits the wall. If the taxi moved successfully two times, it will be in the top-right corner. Otherwise, if the taxi only moved once, it should go back left, and then it will be in the top-left corner.

Using this sequence at the start of the game, we can determine precisely where the taxi will start from. Then, we can set a sequence of actions that will lead the taxi to the passenger, and then to the destination, from the top-left or top-right corner.

## Q-Learning

A Q-Learning algorithm is available. To run it, use the following command:

```bash
python Q-Learning/main.py
```

If you want to customize the hyperparameters, you can use the `--training` and `--alpha` arguments:

```bash
python Q-Learning/main.py --training 1000 --alpha 0.1
```

For a complete list of arguments, use the `--help` flag:

```bash
python Q-Learning/main.py --help
```

>Q-learning lets the agent use the environment's rewards to learn, over time, the best action to take in a given state. We have a reward table, P, that the agent will learn from. It does thing by taking an action in the current state, then updating a Q-value to remember if that action was beneficial.
>
>The values stored in the Q-table are called Q-values, and they map to a (state, action) combination.
>
>A Q-value for a particular state-action combination is representative of the "quality" of an action taken from that state. Better Q-values imply better chances of getting greater rewards.

### SARSA

A SARSA algorithm is available. To run it, use the following command:

```bash
python Sarsa/main.py
```

If you want to customize the hyperparameters, you can use the `--training` and `--alpha` arguments:

```bash
python Sarsa/main.py --training 1000 --alpha 0.1
```

For a complete list of arguments, use the `--help` flag:

```bash
python Sarsa/main.py --help
```

>SARSA is another reinforcement learning algorithm that enables the agent to learn from the environment's rewards to determine the best action to take in a given state. Similar to Q-learning, SARSA utilizes a reward table, typically denoted as P, for learning. However, SARSA updates its Q-values differently.
>
>The Q-values in SARSA represent the quality of taking a specific action from a particular state. These values are stored in a Q-table and are updated based on the agent's experiences. Unlike Q-learning, where the Q-value update is based on the maximum Q-value of the next state, SARSA updates its Q-values based on the action actually taken in the next state, hence its name "State-Action-Reward-State-Action".
>
>In SARSA, the agent selects an action based on its current policy, takes that action in the current state, observes the reward, transitions to the next state, selects the next action based on the policy, and finally updates the Q-value of the previous state-action pair. This process allows SARSA to learn directly from its interactions with the environment and adjust its actions accordingly.

## Resources

### Hyperparameters tuning

The `optuna` library is used to optimize the hyperparameters of the algorithms used in this project.

It's principle is based on Bayesian Optimization techniques that rationally explore search space, focusing on promising regions and reducing computational ahead.

With the advantages of prioritizing promising regions of the hyperparameter space to reduce the numbers of model evaluation required, Optuna outmarch Grid Search and Random Search. It is a preferred choice for complex scale optimization problems because of it's high efficiency and fast convergence.

You can use the tuner script `Taxi/tuner.py` to find the best hyperparameters for your algorithm.

#### Performance calculation

The optuna library needs a performance calculation function to evaluate the hyperparameters. This function must return a score that will be used to optimize the hyperparameters.

In this project we use the following performance calculation function:

```python
ObjectiveScore = (
  adjusted_success_rate
  + adjusted_mean_rewards_per_steps
  - adjusted_mean_penalties_per_episode
) * success_rate_penalty_factor - worst_episode_penalty
```

Where:

- `adjusted_value` is `value * value_weight`
- `worst_episode_penalty` is `worst_episode.epochs / best_episode.epochs`
- `success_rate_penalty_factor` is `0.5` if success_rate is less than **100%**, otherwise `1`

This score is calculated on a testing batch of episodes.

#### Dashboard

To visualize the optimization process and results, you can use the `optuna-dashboard` command:

```bash
optuna-dashboard sqlite:///tuning.sqlite3
```

### Taxi game

- [Taxi-v3 game](https://gymnasium.farama.org/environments/toy_text/taxi/)

The Taxi Problem involves navigating to passengers in a grid world, picking them up and dropping them off at one of four locations.

There are **four designated pick-up and drop-off locations** (*Red, Green, Yellow and Blue*) in the **5x5 grid world**. The taxi starts off at a random square and the passenger at one of the designated locations.

The goal is **move** the taxi to the **passenger's location**, **pick up the passenger**, **move** to the passenger's **desired destination**, and **drop off the passenger**. Once the passenger is dropped off, the `episode ends`.

The player receives `positive rewards` for successfully dropping-off the passenger at the correct location. `Negative rewards` for incorrect attempts to pick-up/drop-off passenger and for each step where another reward is not received.
