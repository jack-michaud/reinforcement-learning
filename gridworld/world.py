from dataclasses import dataclass
import enum
from typing import Callable


@dataclass
class Coordinate:
    x: int
    y: int


class Direction(enum.Enum):
    UP = (0, -1)
    DOWN = (0, +1)
    LEFT = (-1, 0)
    RIGHT = (+1, 0)


class World:
    """
    A basic gridworld. Provide a dimension of the world and a function that returns the rewards for an x,y coordinate.
    """

    current_position: Coordinate
    x_len: int
    y_len: int

    def __init__(
        self,
        x_len: int,
        y_len: int,
        reward_function: Callable[[Coordinate], float],
        initial_position: Coordinate,
    ):
        self.current_position = initial_position
        self.x_len = x_len
        self.y_len = y_len
        self.cells = [
            [reward_function(Coordinate(x, y)) for x in range(x_len)]
            for y in range(y_len)
        ]

    def move(self, direction: Direction) -> float:
        x_offset, y_offset = direction.value
        self.current_position.x = min(
            self.x_len - 1, max(0, self.current_position.x + x_offset)
        )
        self.current_position.y = min(
            self.y_len - 1, max(0, self.current_position.y + y_offset)
        )
        return self.cells[self.current_position.x][self.current_position.y]

    def print(self):
        for row in range(self.y_len):
            for column in range(self.x_len):
                if self.current_position.x == column and self.current_position.y == row:
                    print("X", end="")
                else:
                    print(".", end="")
            print("\n", end="")


if __name__ == "__main__":
    world = World(2, 2, lambda x: -1, initial_position=Coordinate(1, 1))
    world.print()
    world.move(Direction.DOWN)
    world.print()
    world.move(Direction.UP)
    world.print()
    world.move(Direction.UP)
    world.print()
    world.move(Direction.LEFT)
    world.print()
