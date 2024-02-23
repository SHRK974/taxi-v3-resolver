# T-AIA-902 - Taxi Driver

## Project description

The goal of this project is to solve the Taxi-v3 game with an optimized model-free episodic learning algorithm. The agent is a taxi that must pick up some random passengers, and drop them off at some specific locations. The game environment can be found in the Gym library.

> You are free to choose the algorithm, whether on-policy or off-policy.
> We recommand Deep Q-Learning, or Monte Carlo-based algorithms, but feel free to find the best possible approach.

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

### Bruteforce

A naive bruteforce algorithm is available to compare with your optimized algorithm. To run it, use the following command:

```bash
python Bruteforce/main.py --episodes 1000
```

> The `--episodes` argument is optional and defaults to 100. You are free to change it to any value you want.

A benchmark and a report is available in the `Bruteforce` folder.

## Resources

### Taxi game

- [Taxi-v3 game](https://gymnasium.farama.org/environments/toy_text/taxi/)

The Taxi Problem involves navigating to passengers in a grid world, picking them up and dropping them off at one of four locations.

There are **four designated pick-up and drop-off locations** (*Red, Green, Yellow and Blue*) in the **5x5 grid world**. The taxi starts off at a random square and the passenger at one of the designated locations.

The goal is **move** the taxi to the **passenger's location**, **pick up the passenger**, **move** to the passenger's **desired destination**, and **drop off the passenger**. Once the passenger is dropped off, the `episode ends`.

The player receives `positive rewards` for successfully dropping-off the passenger at the correct location. `Negative rewards` for incorrect attempts to pick-up/drop-off passenger and for each step where another reward is not received.
