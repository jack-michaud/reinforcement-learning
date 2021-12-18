import numpy as np


class EpsilonGreedySampleAveragesAgent:
    def __init__(self, levers, epsilon: float):
        self.epsilon = epsilon
        self.sample_averages = {
            int(i): {"count": 0, "expected_value": 0} for i in range(levers)
        }
        self.steps_taken = 0

    def step(self, bandit):
        action = None
        if np.random.rand() < self.epsilon:
            # Choose a random action
            action = np.random.choice(list(self.sample_averages.keys()))
        else:
            # Choose the greedy action
            greedy_action = None
            greedy_action_value = -1
            for action, action_state in self.sample_averages.items():
                if action_state["expected_value"] > greedy_action_value:
                    greedy_action_value = action_state["expected_value"]
                    greedy_action = action

            action = greedy_action

        # Pull the lever and get the reward
        reward = bandit[action][1]()

        # Update the sample averages state using the update rule
        self.sample_averages[action]["count"] += 1
        current_expected_value = self.sample_averages[action]["expected_value"]
        step_size = 1 / self.sample_averages[action]["count"]
        self.sample_averages[action][
            "expected_value"
        ] = current_expected_value + step_size * (reward - current_expected_value)

        return reward, action


class SoftmaxAgent:
    def __init__(self, levers, temperature: float = 1):
        self.sample_averages = {
            int(i): {"count": 0, "expected_value": 0} for i in range(levers)
        }
        self.steps_taken = 0
        self.temperature = temperature

    def step(self, bandit):
        action = None
        # Use softmax action selection

        # Get probabilities of each action
        divisor = np.sum(
            np.exp(
                np.array(
                    list(
                        map(
                            lambda average: average["expected_value"]
                            / self.temperature,
                            self.sample_averages.values(),
                        )
                    )
                )
            )
        )
        probabilities = [
            np.exp(average["expected_value"] / self.temperature) / divisor
            for average in self.sample_averages.values()
        ]

        action = np.random.choice(list(self.sample_averages.keys()), p=probabilities)

        # Pull the lever and get the reward
        reward = bandit[action][1]()

        # Update the sample averages state using the update rule
        self.sample_averages[action]["count"] += 1
        current_expected_value = self.sample_averages[action]["expected_value"]
        step_size = 1 / self.sample_averages[action]["count"]
        self.sample_averages[action][
            "expected_value"
        ] = current_expected_value + step_size * (reward - current_expected_value)

        return reward, action


class EpsilonGreedyConstantStepSize:
    def __init__(self, levers, epsilon: float, alpha: float):
        self.epsilon = epsilon
        self.alpha = alpha
        self.sample_averages = {int(i): {"expected_value": 0} for i in range(levers)}

    def step(self, bandit):
        action = None
        if np.random.rand() < self.epsilon:
            # Choose a random action
            action = np.random.choice(list(self.sample_averages.keys()))
        else:
            # Choose the greedy action
            greedy_action = None
            greedy_action_value = -1
            for action, action_state in self.sample_averages.items():
                if action_state["expected_value"] > greedy_action_value:
                    greedy_action_value = action_state["expected_value"]
                    greedy_action = action

            action = greedy_action

        reward = bandit[action][1]()

        # Update the sample averages state using the update rule with a constant step size
        current_expected_value = self.sample_averages[action]["expected_value"]
        step_size = self.alpha
        self.sample_averages[action][
            "expected_value"
        ] = current_expected_value + step_size * (reward - current_expected_value)

        return reward, action
