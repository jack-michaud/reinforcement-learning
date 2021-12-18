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


def generate_nonstationary_lever():
    """
    Creates a lever; a function that returns a reward.
    The reward returned is a normal distribution around a center with a variance of 1.
    The center of the reward for this lever is initially a constant (5) but will take random walks and change value.
    The center of the reward will only ever be between [0,9].
    """
    action_value = {"value": 0}
    step_size = 2

    def return_reward():
        action_value["value"] = np.random.choice(
            [
                min(9, action_value["value"] + step_size),
                max(0, action_value["value"] - step_size),
                action_value["value"],
            ],
        )
        return max(0, min(10, np.random.normal(action_value["value"], 1)))

    def get_action_value():
        return action_value["value"] + 1

    return get_action_value, return_reward


def generate_nonstationary_bandit(n):
    return [generate_nonstationary_lever() for _ in range(n)]
