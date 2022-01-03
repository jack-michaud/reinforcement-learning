

from .carrentals import Rental, RentalProbability, RentalState, poisson_


def test_poisson():
    for lamb in [1,2,3,4,5,6]:
        p = poisson_(lamb)
        assert 4 in p.probs
        assert abs(1 - sum(p.probs.values())) < 0.01

def test_poisson_values():
    p = poisson_(4)
    assert len(p.probs.keys()) == len(list(p.iterate_possible_values()))


def test_rental_state_transitions():
    rental = Rental(
        RentalProbability(4, 3),
        RentalProbability(3, 2),
    )

    transitions = rental.possible_state_transitions(RentalState(1, 2))
    print(transitions)

