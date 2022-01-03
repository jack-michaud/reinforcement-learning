import math
from typing import Dict, Generator, Optional, Tuple, List
from dataclasses import dataclass
import dataclasses
import numpy as np

def get_poisson_random(lamb):
    """
    Returns a random number from a poisson distribution
    """
    return np.random.poisson(lamb)

def get_poisson_probability(lamb, x) -> float:
    """
    Returns the probability x is returned from a poisson distribution 
    given the expected value lambda.
    """
    return np.exp(-lamb) * (lamb ** x) / math.factorial(x)

class poisson_():
    # Inspired by https://towardsdatascience.com/elucidating-policy-iteration-in-reinforcement-learning-jacks-car-rental-problem-d41b34c8aec7
    min_val: int
    min_val: int
    probs: Dict[int, float]

    def __init__(self, lamb) -> None:
        self.min_val = 0
        self.max_val = float('inf')
        self.probs = {}

        # Minimum probability that we care about
        epsilon = 0.005

        # Find min
        while True:
            probability_min = get_poisson_probability(lamb, self.min_val)
            if probability_min < epsilon:
                self.min_val += 1
            else:
                self.probs[self.min_val] = probability_min
                break
        # Find max
        self.max_val = self.min_val + 1
        while True:
            probability_max = get_poisson_probability(lamb, self.max_val)
            if probability_max > epsilon:
                self.probs[self.max_val] = probability_max
                self.max_val += 1
            else:
                break

    def iterate_possible_values(self) -> Generator[Tuple[int, float], None, None]:
        for val in range(self.min_val, self.max_val):
            yield val, self.probs[val]


def iterate_through_state_space(max_cars):
    for car_count1 in range(max_cars + 1):
        for car_count2 in range(max_cars + 1):
            yield RentalState(
                cars_at_location1=car_count1,
                cars_at_location2=car_count2,
            )

@dataclass
class RentalProbability:
    """
    Stores the probability of a return and a rental as lambda values
    """
    return_probability: int
    rental_probability: int

@dataclass
class RentalState:
    """
    Stores the number of cars in both rental location.
    """
    cars_at_location1: int
    cars_at_location2: int



class Rental:
    """
    A simulation of Jack's multi-location car rental company.
    http://incompleteideas.net/sutton/book/ebook/node43.html (example 4.2)

    - The number of cars requested and returned at each location is a Poisson random variable.
      - Suppose lambda is 3 and 4 for rental requests and 3 and 2 for returns. 
      - The probabilities should be different for both dealerships.
    - If a car is available, Jack is credited $10.
    - Jack can move cars between dealerships at $2/car.
      - A maximum of five cars can be moved from one location to the other in one night
    - There can be no more than 20 cars at a location (any additional cars returned are ignored)

    """
    state: RentalState

    def __init__(self, rental_probability1: RentalProbability, rental_probability2: RentalProbability, max_cars=20, rental_price=10, move_cost=2, initial_state=None):
        """
        Initialize the rental class with the rental probabilities
        """
        self.rental_probability1 = rental_probability1
        self.rental_probability2 = rental_probability2
        self.max_cars = max_cars
        self.rental_price = rental_price
        self.move_cost = move_cost

        if initial_state is None:
            initial_state = RentalState(
                cars_at_location1=0,
                cars_at_location2=0,
            )
        self.state = initial_state

    def get_returns_and_rentals(self, location: int) -> Tuple[int, int]:
        """
        Returns the number of returns and rentals for a given location
        """
        rental_probability = self.rental_probabilities[location]
        return (
            get_poisson_random(rental_probability.return_probability),
            get_poisson_random(rental_probability.rental_probability)
        )

    def update_locations(self) -> int:
        """
        Updates the locations with the total number of cars.
        The total number of cars is clamped between 0 and max_cars.
        Returns the total rental count.
        """
        total_rental_count = 0
        returns = get_poisson_random(self.rental_probability1.return_probability)
        rentals = get_poisson_random(self.rental_probability1.rental_probability)
        total_rental_count += min(rentals, self.state.cars_at_location1)
        self.state.cars_at_location1 = min(self.max_cars, max(0, self.state.cars_at_location1 + returns - rentals))

        returns = get_poisson_random(self.rental_probability2.return_probability)
        rentals = get_poisson_random(self.rental_probability2.rental_probability)
        total_rental_count += min(rentals, self.state.cars_at_location2)
        self.state.cars_at_location2 = min(self.max_cars, max(0, self.state.cars_at_location2 + returns - rentals))

        return total_rental_count

    def possible_state_transitions(self, from_state: RentalState) -> Tuple[RentalState, float, int]:
        """
        Returns a list of possible next states with their probabilities and rewards.
        """
        transitions = []
        for rents1, rent1_probability in poisson_(self.rental_probability1.rental_probability).iterate_possible_values():
            for rents2, rent2_probability in poisson_(self.rental_probability2.rental_probability).iterate_possible_values():
                for returns1, returns1_probability in poisson_(self.rental_probability1.return_probability).iterate_possible_values():
                    for returns2, returns2_probability in poisson_(self.rental_probability2.return_probability).iterate_possible_values():
                        transition_probability = rent1_probability * rent2_probability * returns1_probability * returns2_probability
                        transitions.append((
                            RentalState(
                                min(self.max_cars, max(0, from_state.cars_at_location1 - rents1 + returns1)),
                                min(self.max_cars, max(0, from_state.cars_at_location2 - rents2 + returns2))
                            ),
                            transition_probability,
                            self.rental_price * (min(rents1, from_state.cars_at_location1) + min(rents2, from_state.cars_at_location2))
                        ))
        return transitions

    def take_action(self, state: RentalState, action: int) -> Tuple[int, RentalState]:
        """
        The action represents a move from one location to another.
        A positive number represents moving from location 1 to location 2 and
        a negative number represents moving from location 2 to location 1.

        Returns the reward for taking this action and the new state.
        """
        rewards = 0
        state = RentalState(
            cars_at_location1=state.cars_at_location1,
            cars_at_location2=state.cars_at_location2,
        )
            
        while action != 0:
            if action > 0:
                if state.cars_at_location1 > 0:
                    state.cars_at_location1 -= 1
                    if state.cars_at_location2 < self.max_cars:
                        state.cars_at_location2 += 1
                rewards -= self.move_cost
                action -= 1
            else:
                if state.cars_at_location2 > 0:
                    state.cars_at_location2 -= 1
                    if state.cars_at_location1 < self.max_cars:
                        state.cars_at_location1 += 1
                rewards -= self.move_cost
                action += 1
        return rewards, state



    def step(self, action: int, state_override: Optional[RentalState] = None) -> Tuple[int, RentalState]:
        """
        Simulates a day of rental activity.

        Returns the reward for this step. 
        """
        if state_override:
            self.state = state_override
        rewards, self.state = self.take_action(self.state, action)

        total_rentals = self.update_locations()

        rewards += total_rentals * self.rental_price

        return rewards, self.state


    def print(self):
        """
        Prints the current state of the rental company
        """
        print(f"Location 1: {self.state.cars_at_location1}")
        print(f"Location 2: {self.state.cars_at_location2}")





