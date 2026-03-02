"""
game/snake.py — The Snake (pursuer) entity.

The snake is represented as a deque of (row, col) positions.
The head is at index 0. Movement prepends a new head and removes the tail
(unless the snake just ate, in which case the tail stays → growth).

WHY A DEQUE?
    collections.deque gives O(1) appendleft and pop operations,
    compared to O(n) for list.insert(0, ...). This matters when
    we run millions of steps.
"""

from collections import deque
from game.actions import Action, DIRECTION_VECTORS, OPPOSITES


class Snake:
    """
    The pursuer agent in the game.

    Attributes:
        body: deque of (row, col) positions, head is body[0]
        direction: current heading (Action enum)
        alive: whether the snake is still in play
        grow_pending: segments to add on next move(s)
    """

    def __init__(self, start_pos: tuple[int, int], direction: Action = Action.RIGHT):
        self.body: deque[tuple[int, int]] = deque()
        self.body.append(start_pos)
        # Start with 2 extra body segments behind the head
        dr, dc = DIRECTION_VECTORS[OPPOSITES[direction]]
        for i in range(1, 3):
            self.body.append((start_pos[0] + dr * i, start_pos[1] + dc * i))
        self.direction = direction
        self.alive = True
        self.grow_pending = 0

    @property
    def head(self) -> tuple[int, int]:
        """Position of the snake's head."""
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)

    def set_direction(self, action: Action) -> None:
        """
        Update direction, ignoring reversals.

        WHY IGNORE REVERSALS?
            If the snake is moving RIGHT and the agent outputs LEFT,
            the snake would immediately collide with its own body.
            This is a standard Snake game rule to prevent trivial deaths.
        """
        if action in OPPOSITES and OPPOSITES[action] != self.direction:
            self.direction = action
        elif action not in OPPOSITES:
            # STAY is not valid for snake, ignore
            pass

    def move(self) -> tuple[int, int]:
        """
        Move the snake one step in its current direction.

        Returns:
            new_head: the new head position

        The snake moves by prepending a new head. If no growth is pending,
        the tail is removed. If growth IS pending, the tail stays.
        """
        dr, dc = DIRECTION_VECTORS[self.direction]
        new_head = (self.head[0] + dr, self.head[1] + dc)
        self.body.appendleft(new_head)

        if self.grow_pending > 0:
            self.grow_pending -= 1
        else:
            self.body.pop()

        return new_head

    def grow(self, amount: int = 1) -> None:
        """Queue growth segments (added during subsequent moves)."""
        self.grow_pending += amount

    def check_self_collision(self) -> bool:
        """Check if the head occupies the same cell as any body segment."""
        return self.head in list(self.body)[1:]

    def get_body_set(self) -> set[tuple[int, int]]:
        """Return body positions as a set for O(1) collision lookups."""
        return set(self.body)
