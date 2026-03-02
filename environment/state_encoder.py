"""
environment/state_encoder.py — Converts raw game state to tensor observations.

WHY MULTI-CHANNEL GRIDS?
    Think of it like a satellite image with different spectral bands.
    Each channel isolates one type of information:
    - Channel 0: Where is the snake body? (obstacle map)
    - Channel 1: Where is the snake head? (threat location)
    - Channel 2: Where is the bait? (target/self location)
    - Channel 3: Where are the walls? (boundary map)

    CNNs can learn spatial patterns across these channels, like
    "the snake head is 3 cells away and approaching from the left."
"""

import numpy as np
from game.grid import Grid, CellType


class StateEncoder:
    """
    Encodes the game grid into multi-channel tensor observations.

    Output shape: (4, grid_size, grid_size) — 4 channels of binary maps.
    """

    def __init__(self, grid_size: int):
        self.grid_size = grid_size

    def encode_for_snake(self, grid: Grid, snake_body: list,
                          snake_head: tuple, bait_pos: tuple) -> np.ndarray:
        """
        Create the snake's observation.

        The snake sees:
          Ch0: Its own body (to avoid self-collision)
          Ch1: Its own head (self-awareness)
          Ch2: The bait (target to pursue)
          Ch3: Walls (boundaries to avoid)
        """
        obs = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)

        # Channel 0: Snake body
        for r, c in snake_body:
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                obs[0, r, c] = 1.0

        # Channel 1: Snake head
        obs[1, snake_head[0], snake_head[1]] = 1.0

        # Channel 2: Bait position
        obs[2, bait_pos[0], bait_pos[1]] = 1.0

        # Channel 3: Walls
        obs[3] = (grid.cells == CellType.WALL).astype(np.float32)

        return obs

    def encode_for_bait(self, grid: Grid, snake_body: list,
                         snake_head: tuple, bait_pos: tuple) -> np.ndarray:
        """
        Create the bait's observation.

        The bait sees:
          Ch0: Snake body (obstacles to avoid)
          Ch1: Snake head (primary threat)
          Ch2: Own position (self-awareness)
          Ch3: Walls (boundaries — can get cornered!)
        """
        obs = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)

        # Channel 0: Snake body
        for r, c in snake_body:
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                obs[0, r, c] = 1.0

        # Channel 1: Snake head
        obs[1, snake_head[0], snake_head[1]] = 1.0

        # Channel 2: Bait (self) position
        obs[2, bait_pos[0], bait_pos[1]] = 1.0

        # Channel 3: Walls
        obs[3] = (grid.cells == CellType.WALL).astype(np.float32)

        return obs
