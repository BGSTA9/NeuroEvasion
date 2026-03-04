"""
game/engine.py — The core game loop and reward computation.

This is the heart of NeuroEvasion. The engine:
1. Manages the grid, snake, and bait
2. Processes actions from both agents each step
3. Computes rewards based on the outcome
4. Detects terminal conditions (capture, death, timeout)

MULTI-DISCRETE EXTENSION:
    step() now accepts either a plain int (single-discrete, legacy) or a
    MultiDiscreteAction (multi-discrete). The tool component is dispatched
    to _apply_tool() which logs tool choices and — once tool mechanics are
    implemented — applies their in-game effects.

SEPARATION OF CONCERNS:
    The engine knows NOTHING about neural networks, training, or rendering.
    It is pure game logic. This makes it testable, fast, and reusable.
"""

from game.grid import Grid, CellType
from game.snake import Snake
from game.bait import Bait
from game.actions import Action, MultiDiscreteAction, SNAKE_TOOL_LABELS, BAIT_TOOL_LABELS
from config import GameConfig, RewardConfig


class GameEngine:
    """
    Manages a single episode of the NeuroEvasion game.

    Usage:
        engine = GameEngine(game_config, reward_config)
        engine.reset()
        while not engine.done:
            snake_reward, bait_reward, done, info = engine.step(snake_action, bait_action)
    """

    def __init__(self, game_config: GameConfig, reward_config: RewardConfig):
        self.config = game_config
        self.rewards = reward_config
        self.grid = Grid(game_config.grid_size)
        self.snake: Snake | None = None
        self.bait: Bait | None = None
        self.current_step = 0
        self.done = False
        self.winner: str | None = None

    def reset(self) -> dict:
        """
        Reset the game to initial state.

        Snake starts in the left third, bait starts in the right third.
        This ensures they begin with some distance between them.

        Returns:
            info dict with initial positions
        """
        self.grid.reset()
        self.current_step = 0
        self.done = False
        self.winner = None

        size = self.config.grid_size

        # Snake spawns in left portion of the grid
        snake_row = size // 2
        snake_col = size // 4
        self.snake = Snake((snake_row, snake_col), direction=Action.RIGHT)

        # Bait spawns in right portion of the grid
        bait_row = size // 2
        bait_col = 3 * size // 4
        self.bait = Bait((bait_row, bait_col))

        self._update_grid()

        return {
            "snake_pos": self.snake.head,
            "bait_pos": self.bait.position,
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, snake_action: int | MultiDiscreteAction,
             bait_action: int | MultiDiscreteAction) -> tuple[float, float, bool, dict]:
        """
        Execute one game step.

        Accepts either plain int actions (legacy single-discrete) or
        MultiDiscreteAction instances (multi-discrete mode).

        Order of operations (critical for fairness):
            1. Decode actions; resolve tool components
            2. Snake moves (movement component only)
            3. Check if snake died (wall/self collision)
            4. Bait moves (movement component only)
            5. Check if bait was captured
            6. Compute rewards + apply tool effects (stub)

        Args:
            snake_action: int or MultiDiscreteAction for the snake
            bait_action:  int or MultiDiscreteAction for the bait

        Returns:
            (snake_reward, bait_reward, done, info)
        """
        if self.done:
            raise RuntimeError("Game is over. Call reset() to start a new episode.")

        self.current_step += 1
        info = {"event": "step"}

        # ── Decode multi-discrete actions ──────────────────────────────────────
        snake_move_int, snake_tool_int = self._decode_action(snake_action)
        bait_move_int,  bait_tool_int  = self._decode_action(bait_action)

        # Record tool choices in info for logging / analysis
        info["snake_tool"] = SNAKE_TOOL_LABELS.get(snake_tool_int, str(snake_tool_int))
        info["bait_tool"]  = BAIT_TOOL_LABELS.get(bait_tool_int,  str(bait_tool_int))

        # Calculate distance BEFORE moves (for reward shaping)
        dist_before = self._manhattan_distance()

        # --- 1. Snake moves ---
        self.snake.set_direction(Action(snake_move_int))
        new_head = self.snake.move()

        # --- 2. Check snake death ---
        if self.grid.is_wall(*new_head) or self.snake.check_self_collision():
            self.done = True
            self.winner = "bait"
            self.snake.alive = False
            info["event"] = "snake_died"
            self._update_grid()
            return (
                self.rewards.death_penalty,          # Snake: big penalty
                -self.rewards.death_penalty,         # Bait: big reward (zero-sum)
                True,
                info,
            )

        # --- 3. Check immediate capture (snake moved onto bait) ---
        if new_head == self.bait.position:
            self.done = True
            self.winner = "snake"
            self.bait.alive = False
            self.snake.grow()
            info["event"] = "capture"
            self._update_grid()
            return (
                self.rewards.capture_reward,
                -self.rewards.capture_reward,
                True,
                info,
            )

        # --- 4. Bait moves ---
        if self.current_step % self.config.bait_move_every == 0:
            self.bait.move(Action(bait_move_int), self.grid)

        # --- 5. Check capture after bait move (bait moved into snake head) ---
        if self.bait.position == self.snake.head:
            self.done = True
            self.winner = "snake"
            self.bait.alive = False
            info["event"] = "capture"
            self._update_grid()
            return (
                self.rewards.capture_reward,
                -self.rewards.capture_reward,
                True,
                info,
            )

        # --- 6. Check timeout ---
        if self.current_step >= self.config.max_steps:
            self.done = True
            self.winner = "bait"  # Bait survived!
            info["event"] = "timeout"
            self._update_grid()
            return (
                self.rewards.timeout_penalty_snake,
                self.rewards.timeout_reward_bait,
                True,
                info,
            )

        # --- 7. Apply tool effects (extensible stubs) ---
        tool_snake_bonus, tool_bait_bonus = self._apply_tools(
            snake_tool_int, bait_tool_int, info
        )

        # --- 8. Compute shaping rewards ---
        dist_after = self._manhattan_distance()

        if dist_after < dist_before:
            # Snake got closer → reward snake, penalize bait
            snake_reward = self.rewards.distance_reward + self.rewards.step_penalty_snake
            bait_reward  = -self.rewards.distance_reward + self.rewards.step_reward_bait
        elif dist_after > dist_before:
            # Bait escaped further → reward bait, penalize snake
            snake_reward = -self.rewards.distance_reward + self.rewards.step_penalty_snake
            bait_reward  =  self.rewards.distance_reward + self.rewards.step_reward_bait
        else:
            snake_reward = self.rewards.step_penalty_snake
            bait_reward  = self.rewards.step_reward_bait

        # Incorporate any tool bonuses/penalties
        snake_reward += tool_snake_bonus
        bait_reward  += tool_bait_bonus

        info["distance"] = dist_after
        info["step"] = self.current_step

        self._update_grid()
        return (snake_reward, bait_reward, False, info)

    # ──────────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _decode_action(action: int | MultiDiscreteAction) -> tuple[int, int]:
        """
        Return (move_int, tool_int) regardless of action format.

        Plain int → (action, 0)  [tool NONE, fully backward-compatible]
        MultiDiscreteAction → (move, tool)
        """
        if isinstance(action, MultiDiscreteAction):
            return action.move, action.tool
        return int(action), 0  # legacy: tool = NONE

    def _apply_tools(
        self,
        snake_tool: int,
        bait_tool: int,
        info: dict,
    ) -> tuple[float, float]:
        """
        Dispatch tool-use sub-actions to their effect handlers.

        Returns:
            (snake_bonus_reward, bait_bonus_reward)

        EXTENSIBILITY:
            Add new tools by inserting into the if/elif chains below.
            Each tool returns a float bonus/penalty that is added to the
            agent's shaped reward for this step.

        CURRENT STATUS: All tools are stubs (return 0.0). Their selection
        is recorded in `info` for trajectory analysis; mechanical effects
        will be implemented per-tool in follow-up sprints.
        """
        snake_bonus = self._apply_snake_tool(snake_tool, info)
        bait_bonus  = self._apply_bait_tool(bait_tool, info)
        return snake_bonus, bait_bonus

    def _apply_snake_tool(self, tool: int, info: dict) -> float:
        """
        Apply a snake tool effect. Returns reward bonus for this step.

        Tool registry:
            0 (NONE)  — no effect
            1 (DASH)  — [stub] sprint move; cost: small energy penalty
            2 (SLOW)  — [stub] slow bait; cost: small energy penalty
        """
        if tool == 0:    # NONE
            return 0.0
        elif tool == 1:  # DASH — stub
            info["snake_tool_active"] = "DASH"
            return 0.0   # placeholder: will return speed bonus once implemented
        elif tool == 2:  # SLOW — stub
            info["snake_tool_active"] = "SLOW"
            return 0.0   # placeholder: will penalise bait movement subtraction
        return 0.0       # unknown tool → no-op

    def _apply_bait_tool(self, tool: int, info: dict) -> float:
        """
        Apply a bait tool effect. Returns reward bonus for this step.

        Tool registry:
            0 (NONE)  — no effect
            1 (BLINK) — [stub] teleport one safe cell; cost: small energy penalty
            2 (DECOY) — [stub] place ghost marker; cost: small energy penalty
        """
        if tool == 0:    # NONE
            return 0.0
        elif tool == 1:  # BLINK — stub
            info["bait_tool_active"] = "BLINK"
            return 0.0   # placeholder: will apply position shift once implemented
        elif tool == 2:  # DECOY — stub
            info["bait_tool_active"] = "DECOY"
            return 0.0   # placeholder: will add ghost channel to obs once implemented
        return 0.0       # unknown tool → no-op

    def _manhattan_distance(self) -> int:
        """Manhattan distance between snake head and bait."""
        return abs(self.snake.head[0] - self.bait.row) + abs(self.snake.head[1] - self.bait.col)

    def _update_grid(self) -> None:
        """Redraw all entities onto the grid."""
        self.grid.reset()

        if self.snake and self.snake.alive:
            for pos in self.snake.body:
                r, c = pos
                if 0 <= r < self.grid.size and 0 <= c < self.grid.size:
                    self.grid.set_cell(r, c, CellType.SNAKE_BODY)
            hr, hc = self.snake.head
            if 0 <= hr < self.grid.size and 0 <= hc < self.grid.size:
                self.grid.set_cell(hr, hc, CellType.SNAKE_HEAD)

        if self.bait and self.bait.alive:
            self.grid.set_cell(self.bait.row, self.bait.col, CellType.BAIT)

    def get_state(self) -> dict:
        """Return the full game state as a dictionary."""
        return {
            "grid":       self.grid.cells.copy(),
            "snake_head": self.snake.head if self.snake else None,
            "snake_body": list(self.snake.body) if self.snake else [],
            "bait_pos":   self.bait.position if self.bait else None,
            "step":       self.current_step,
            "done":       self.done,
            "winner":     self.winner,
        }
