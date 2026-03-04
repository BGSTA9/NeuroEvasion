"""
game/actions.py — Action definitions and direction mappings.

DESIGN NOTE:
    Actions are integers for neural network compatibility (output index).
    Direction vectors are (row_delta, col_delta) tuples.

MULTI-DISCRETE EXTENSION:
    Each agent can now select two independent sub-actions each step:
        1. A movement action  (UP / DOWN / LEFT / RIGHT / STAY)
        2. A tool-use action  (NONE / agent-specific tools)

    The two sub-actions are encoded into a single integer for replay buffer
    compatibility via MultiDiscreteAction.encode() / .decode().

    Tool semantic effects are dispatched by the GameEngine._apply_tool().
    Initial tool implementations are stubs; the action-space wiring is live.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum


# ──────────────────────────────────────────────────────────────────────────────
#  Movement actions (same as before — both agents share this enum)
# ──────────────────────────────────────────────────────────────────────────────

class Action(IntEnum):
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3
    STAY  = 4     # Only available to bait


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
    Action.UP:    Action.DOWN,
    Action.DOWN:  Action.UP,
    Action.LEFT:  Action.RIGHT,
    Action.RIGHT: Action.LEFT,
}


# ──────────────────────────────────────────────────────────────────────────────
#  Tool-use actions
# ──────────────────────────────────────────────────────────────────────────────

class SnakeTool(IntEnum):
    """
    Offensive / tactical tools available to the Snake.

    NONE  — no tool used this step (always valid)
    DASH  — [stub] move two cells in one step, breaking reaction time
    SLOW  — [stub] temporarily reduce bait's move frequency for N steps
    """
    NONE = 0
    DASH = 1
    SLOW = 2


class BaitTool(IntEnum):
    """
    Evasion / defensive tools available to the Bait.

    NONE   — no tool used this step (always valid)
    BLINK  — [stub] teleport one cell in a random safe direction
    DECOY  — [stub] leave a temporary ghost mark that confuses the snake's observation
    """
    NONE  = 0
    BLINK = 1
    DECOY = 2


# Human-readable labels for logging / analysis
SNAKE_TOOL_LABELS = {t.value: t.name for t in SnakeTool}
BAIT_TOOL_LABELS  = {t.value: t.name for t in BaitTool}


# ──────────────────────────────────────────────────────────────────────────────
#  Multi-Discrete action container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MultiDiscreteAction:
    """
    A compound action: (movement, tool-use).

    WHY ENCODE INTO A SINGLE INT?
        The existing ReplayBuffer stores actions as numpy int arrays.
        Rather than rewriting the buffer, we pack both sub-actions into one
        integer using a simple mixed-radix encoding:

            encoded = move * num_tool_actions + tool

        This is lossless as long as 0 <= tool < num_tool_actions.

    Args:
        move: Movement sub-action (from Action enum).
        tool: Tool-use sub-action (from SnakeTool or BaitTool enum).
    """
    move: int
    tool: int

    # ── Codec ──────────────────────────────────────────────────────────────────

    def encode(self, num_tool_actions: int) -> int:
        """Pack (move, tool) into a single integer for replay buffer storage."""
        return self.move * num_tool_actions + self.tool

    @staticmethod
    def decode(encoded: int, num_tool_actions: int) -> "MultiDiscreteAction":
        """Unpack a single integer back into a MultiDiscreteAction."""
        move = encoded // num_tool_actions
        tool = encoded  % num_tool_actions
        return MultiDiscreteAction(move=move, tool=tool)

    def __repr__(self) -> str:
        return f"MultiDiscreteAction(move={self.move}, tool={self.tool})"
