import numpy as np


def generate_lever():
    """
    Creates a lever; a function that returns a reward.
    The reward is a normal distribution with variance 1 centered around a random integer between [0,9].
    """
    action_value = int(np.random.rand() * 10)

    def return_reward():
        return np.max([0, np.random.normal(action_value, 1)])

    def get_action_value():
        return action_value + 1

    return get_action_value, return_reward


def generate_bandit(n):
    return [generate_lever() for _ in range(n)]


