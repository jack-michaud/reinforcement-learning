import numpy as np
from .bandit import generate_bandit
from .agent import SoftmaxAgent


def main():
    n = 10
    bandit = generate_bandit(n)
    agent = SoftmaxAgent(n)
    for _ in range(2000):
        agent.step(bandit=bandit)
    print(agent.sample_averages)


if __name__ == "__main__":
    main()
