"""
game/actions.py — Action definitions and direction mappings.

DESIGN NOTE:
    Actions are integers for neural network compatibility (output index).
    Direction vectors are (row_delta, col_delta) tuples.
"""

from enum import IntEnum


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4     # Only available to bait


# Maps action → (row_delta, col_delta)
DIRECTION_VECTORS = {
    Action.UP:    (-1,  0),
    Action.DOWN:  ( 1,  0),
    Action.LEFT:  ( 0, -1),
    Action.RIGHT: ( 0,  1),
    Action.STAY:  ( 0,  0),
}

# Opposite actions — snake cannot reverse direction
OPPOSITES = {
    Action.UP: Action.DOWN,
    Action.DOWN: Action.UP,
    Action.LEFT: Action.RIGHT,
    Action.RIGHT: Action.LEFT,
}
