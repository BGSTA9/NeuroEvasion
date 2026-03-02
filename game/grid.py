"""
game/grid.py — The grid world data structure.

The grid is an N×N numpy array where each cell contains an integer
representing its contents. Walls form the border.

COORDINATE SYSTEM:
    (0,0) is top-left.
    row increases downward (y-axis).
    col increases rightward (x-axis).
"""

import numpy as np
from enum import IntEnum


class CellType(IntEnum):
    """What a grid cell contains."""
    EMPTY = 0
    WALL = 1
    SNAKE_HEAD = 2
    SNAKE_BODY = 3
    BAIT = 4


class Grid:
    """
    N×N grid world with wall borders.

    The playable area is (N-2)×(N-2) — the outermost ring is walls.

    Example for N=6:
        W W W W W W
        W . . . . W
        W . . . . W
        W . . . . W
        W . . . . W
        W W W W W W
    """

    def __init__(self, size: int = 20):
        self.size = size
        self.cells = np.zeros((size, size), dtype=np.int8)
        self._build_walls()

    def _build_walls(self) -> None:
        """Place walls along all four borders."""
        self.cells[0, :] = CellType.WALL       # Top row
        self.cells[-1, :] = CellType.WALL      # Bottom row
        self.cells[:, 0] = CellType.WALL       # Left column
        self.cells[:, -1] = CellType.WALL      # Right column

    def reset(self) -> None:
        """Clear the grid and rebuild walls."""
        self.cells.fill(CellType.EMPTY)
        self._build_walls()

    def is_wall(self, row: int, col: int) -> bool:
        """Check if a position is a wall."""
        return self.cells[row, col] == CellType.WALL

    def is_empty(self, row: int, col: int) -> bool:
        """Check if a position is empty (safe to move into)."""
        return self.cells[row, col] == CellType.EMPTY

    def set_cell(self, row: int, col: int, cell_type: CellType) -> None:
        """Set the contents of a cell."""
        self.cells[row, col] = cell_type

    def get_playable_positions(self) -> list[tuple[int, int]]:
        """Return all positions inside the walls."""
        positions = []
        for r in range(1, self.size - 1):
            for c in range(1, self.size - 1):
                if self.cells[r, c] == CellType.EMPTY:
                    positions.append((r, c))
        return positions

    def __repr__(self) -> str:
        symbols = {0: "·", 1: "█", 2: "S", 3: "s", 4: "B"}
        rows = []
        for r in range(self.size):
            row_str = " ".join(symbols.get(int(c), "?") for c in self.cells[r])
            rows.append(row_str)
        return "\n".join(rows)
