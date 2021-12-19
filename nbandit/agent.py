from abc import abstractmethod
from typing import Any, Callable, Dict, List, Tuple
import numpy as np


def get_greedy_action(action_values: Dict[int, Any]) -> int:
    greedy_actions = []
    greedy_action_value = -1
    for action, action_state in action_values.items():
        if action_state["expected_value"] > greedy_action_value:
            greedy_action_value = action_state["expected_value"]
            greedy_actions = [action]
        elif action_state["expected_value"] == greedy_action_value:
            greedy_actions.append(action)

    return np.random.choice(greedy_actions)


class Agent:
    @abstractmethod
    def choose_action(self) -> int:
        pass

    @abstractmethod
    def update_state(self, reward: float, action: int):
        pass

    def step(self, bandit: List[Tuple[Callable[[], int], Callable[[], int]]]):
        action = self.choose_action()

        # Pull the lever and get the reward
        reward = bandit[action][1]()

        self.update_state(reward, action)

        return reward, action


class EpsilonGreedySampleAveragesAgent(Agent):
    """
    An action value agent that uses the sample averages method of action value estimation with epsilon greedy action selection.
    """

    def __init__(self, levers, epsilon: float, initial_action_value: float = 0):
        self.epsilon = epsilon
        self.action_values = {
            int(i): {"count": 0, "expected_value": initial_action_value}
            for i in range(levers)
        }

    def choose_action(self):
        action = None
        if np.random.rand() < self.epsilon:
            # Choose a random action
            action = np.random.choice(list(self.action_values.keys()))
        else:
            # Choose the greedy action
            action = get_greedy_action(self.action_values)

        return action

    def update_state(self, reward, action):
        # Update the sample averages state using the update rule
        self.action_values[action]["count"] += 1
        current_expected_value = self.action_values[action]["expected_value"]
        step_size = 1 / self.action_values[action]["count"]
        self.action_values[action][
            "expected_value"
        ] = current_expected_value + step_size * (reward - current_expected_value)


class SoftmaxAgent(Agent):
    """
    An action value agent using sample averages with softmax action selection.
    """

    def __init__(self, levers, temperature: float = 1):
        self.action_values = {
            int(i): {"count": 0, "expected_value": 0} for i in range(levers)
        }
        self.temperature = temperature

    def choose_action(self):
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
                            self.action_values.values(),
                        )
                    )
                )
            )
        )
        probabilities = [
            np.exp(average["expected_value"] / self.temperature) / divisor
            for average in self.action_values.values()
        ]

        action = np.random.choice(list(self.action_values.keys()), p=probabilities)
        return action

    def update_state(self, reward, action):
        # Update the sample averages state using the update rule
        self.action_values[action]["count"] += 1
        current_expected_value = self.action_values[action]["expected_value"]
        step_size = 1 / self.action_values[action]["count"]
        self.action_values[action][
            "expected_value"
        ] = current_expected_value + step_size * (reward - current_expected_value)


class EpsilonGreedyConstantStepSize(Agent):
    """
    An action value method agent that uses a constant step size (rather than the sample averages' 1/k step size), suitable for nonstationary problems.
    """

    def __init__(
        self, levers, epsilon: float, alpha: float, initial_action_value: float = 0
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.action_values = {
            int(i): {"expected_value": initial_action_value} for i in range(levers)
        }

    def choose_action(self):
        action = None
        if np.random.rand() < self.epsilon:
            # Choose a random action
            action = np.random.choice(list(self.action_values.keys()))
        else:
            # Choose the greedy action
            action = get_greedy_action(self.action_values)

        return action

    def update_state(self, reward, action):
        # Update the sample averages state using the update rule with a constant step size
        current_expected_value = self.action_values[action]["expected_value"]
        step_size = self.alpha
        self.action_values[action][
            "expected_value"
        ] = current_expected_value + step_size * (reward - current_expected_value)


class ReinforcementComparisonAgent(Agent):
    """
    A reinforcement comparison method agent. This agent
    keeps track of an average reference reward and updates
    action preferences based on this average.
    """

    def __init__(
        self,
        lever_count: int,
        alpha: float = 0.9,
        beta: float = 0.9,
        initial_preferences_value: float = 0,
        initial_reference_reward: float = 0,
        balance_initial_action_selection: bool = False,
    ):
        """
        Args:
            initial_preferences_value: should encourage exploration
            balance_initial_action_selection: add a factor of (1 - \pi(a)) to the action preference update to prevent early rewards' actions from overpowering other actions. In practice, this doesn't improve things.
        """
        self.lever_count = lever_count
        self.alpha = alpha
        self.beta = beta
        self.balance_initial_action_selection = balance_initial_action_selection
        self.action_preferences = {
            int(idx): initial_preferences_value for idx in range(lever_count)
        }
        self.reference_reward = initial_reference_reward

    def choose_action(self) -> int:
        """
        Use softmax selection of action preferences.
        """
        probabilities = self.get_probability_of_actions()

        return np.random.choice(list(range(self.lever_count)), p=probabilities)

    def get_probability_of_actions(self) -> List[float]:
        action_preferences = np.array(list(self.action_preferences.values()))
        probabilities = np.exp(action_preferences) / np.sum(np.exp(action_preferences))
        return probabilities

    def update_state(self, reward: float, action: int):
        # update action preferences
        if self.balance_initial_action_selection:
            self.action_preferences[action] = self.action_preferences[action] + (
                self.beta
                * (reward - self.reference_reward)
                * self.get_probability_of_actions()[action]
            )
        else:
            self.action_preferences[action] = self.action_preferences[action] + (
                self.beta * (reward - self.reference_reward)
            )

        # update reference reward
        self.reference_reward = self.reference_reward + (
            self.alpha * (reward - self.reference_reward)
        )


class PursuitAgent(Agent):
    def __init__(self, lever_count: int, beta: float = 0.1) -> None:
        self.lever_count = lever_count
        self.beta = beta
        self.action_preferences = {idx: 1 / lever_count for idx in range(lever_count)}
        self.action_values = {
            idx: {"count": 0, "expected_value": 0} for idx in range(lever_count)
        }

    @abstractmethod
    def get_probability_of_actions(self) -> np.ndarray:
        pass

    def choose_action(self) -> int:
        action_probabilities = self.get_probability_of_actions()
        return np.random.choice(list(range(self.lever_count)), p=action_probabilities)

    def update_state(self, reward: float, action: int):
        # Update action value estimate with sample averages
        self.action_values[action]["count"] += 1
        self.action_values[action]["expected_value"] += (
            reward - self.action_values[action]["expected_value"]
        ) / self.action_values[action]["count"]

        # Update the action preferences
        optimal_action = get_greedy_action(self.action_values)
        for idx in range(self.lever_count):
            # If this is the optimal action, weight it closer to 1 by a factor of beta
            # If this is not the optimal action, weight it closer to 0 by a factor of beta
            if idx == optimal_action:
                self.action_preferences[idx] += self.beta * (
                    1 - self.action_preferences[idx]
                )
            else:
                self.action_preferences[idx] -= self.beta * self.action_preferences[idx]
                self.action_preferences[idx] = max(self.action_preferences[idx], 0)


class PursuitAgentSimplePolicy(PursuitAgent):
    def get_probability_of_actions(self) -> np.ndarray:
        action_preferences = np.array(
            [value for value in self.action_preferences.values()]
        )
        return action_preferences


class SoftmaxPursuitAgent(PursuitAgent):
    def get_probability_of_actions(self) -> np.ndarray:
        action_preferences = np.array(
            [value for value in self.action_preferences.values()]
        )
        action_preferences_exp = np.exp(action_preferences - np.max(action_preferences))
        return action_preferences_exp / np.sum(action_preferences_exp)
