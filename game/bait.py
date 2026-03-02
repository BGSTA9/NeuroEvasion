"""
game/bait.py — The Bait (evader) entity.

Unlike the snake, the bait is a single cell that moves freely.
It is the intelligent agent that must learn to survive.

DESIGN PHILOSOPHY:
    The bait is the "star" of NeuroEvasion. While classic Snake treats
    food as a passive object, our bait is a rational agent with its own
    neural network, observations, and learned policy.
"""

from game.actions import Action, DIRECTION_VECTORS


class Bait:
    """
    The evader agent in the game.

    Attributes:
        position: (row, col) current position
        alive: always True until captured
        steps_survived: counter for survival duration
    """

    def __init__(self, start_pos: tuple[int, int]):
        self.position = start_pos
        self.alive = True
        self.steps_survived = 0

    @property
    def row(self) -> int:
        return self.position[0]

    @property
    def col(self) -> int:
        return self.position[1]

    def move(self, action: Action, grid) -> tuple[int, int]:
        """
        Move the bait, respecting walls.

        If the bait would move into a wall, it stays in place.
        This is an important design choice: the bait is "cornered"
        by walls, creating strategic depth.

        Args:
            action: direction to move (or STAY)
            grid: the Grid object for wall checking

        Returns:
            new_position: where the bait ended up
        """
        dr, dc = DIRECTION_VECTORS[action]
        new_row = self.row + dr
        new_col = self.col + dc

        if not grid.is_wall(new_row, new_col):
            self.position = (new_row, new_col)

        self.steps_survived += 1
        return self.position
