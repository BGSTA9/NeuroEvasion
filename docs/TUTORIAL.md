<![CDATA[# 🧠 NeuroEvasion — Complete Implementation Tutorial

> **A step-by-step guide to building a co-evolutionary pursuit-evasion game powered by Deep Neural Networks.**
> 
> *Designed for university students learning Deep Learning & Deep Neural Networks.*

---

## Table of Contents

| Part | Topic | Key Concepts |
|------|-------|-------------|
| **Phase 0** | [Project Setup](#phase-0--project-setup) | Environment, dependencies, configuration |
| **Phase 1** | [Game Engine](#phase-1--the-game-engine) | Grid world, entities, game loop, rewards |
| **Phase 2** | [RL Environment](#phase-2--observation--environment) | State encoding, tensors, Gym API |
| **Phase 3** | [Neural Network Agents](#phase-3--neural-network-agents) | DQN, replay buffer, ε-greedy |
| **Phase 4** | [Training & Co-Evolution](#phase-4--training--co-evolution) | Training loop, self-play, logging |
| **Phase 5** | [Visualization & Evaluation](#phase-5--visualization--evaluation) | Pygame renderer, metrics, demo |

---

## Prerequisites

Before starting, students should understand:
- **Python** — classes, decorators, type hints, dataclasses
- **NumPy** — array manipulation, broadcasting
- **PyTorch basics** — tensors, `nn.Module`, optimizers, loss functions
- **Reinforcement Learning fundamentals** — MDP, reward, policy, value functions

---

## Phase 0 — Project Setup

### Step 0.1 — Create the Project

```bash
# Create project directory and enter it
mkdir NeuroEvasion && cd NeuroEvasion

# Initialize git
git init

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### Step 0.2 — Install Dependencies

Create `requirements.txt`:

```txt
# Core
torch>=2.0.0
numpy>=1.24.0
pygame>=2.5.0

# Training & Monitoring
tensorboard>=2.14.0
matplotlib>=3.7.0

# Development
pytest>=7.4.0
ruff>=0.1.0
black>=23.0.0
```

```bash
pip install -r requirements.txt
```

### Step 0.3 — Create Directory Structure

```bash
mkdir -p game environment agents training visualization evaluation checkpoints logs tests docs
touch game/__init__.py environment/__init__.py agents/__init__.py
touch training/__init__.py visualization/__init__.py evaluation/__init__.py
touch checkpoints/.gitkeep logs/.gitkeep
```

### Step 0.4 — Configuration System

Create `config.py` — **this is where every hyperparameter lives**:

```python
"""
config.py — Central configuration for the NeuroEvasion project.

WHY A DATACLASS?
    Dataclasses give us type-checked, documented, immutable configuration.
    Every "magic number" in the project traces back to here.
"""

from dataclasses import dataclass, field


@dataclass
class GameConfig:
    """Configuration for the game engine."""
    grid_size: int = 20              # N×N grid (includes walls)
    max_steps: int = 200             # Max steps per episode before timeout
    bait_move_every: int = 1         # Bait moves once per k snake moves
    bait_can_stay: bool = True       # Allow bait to choose "stay" action


@dataclass
class RewardConfig:
    """
    Reward shaping for the zero-sum game.
    
    DESIGN PRINCIPLE: Snake reward = -Bait reward (zero-sum).
    Shaping rewards guide learning when terminal rewards are sparse.
    """
    capture_reward: float = 10.0       # Snake captures bait
    death_penalty: float = -10.0       # Snake hits wall/self
    step_penalty_snake: float = -0.01  # Urgency: snake must act fast
    step_reward_bait: float = 0.01     # Bait rewarded for surviving
    distance_reward: float = 0.1       # Reward for closing/opening distance
    timeout_penalty_snake: float = -1.0
    timeout_reward_bait: float = 5.0


@dataclass
class AgentConfig:
    """Configuration for the DQN agents."""
    learning_rate: float = 1e-4
    gamma: float = 0.99              # Discount factor
    epsilon_start: float = 1.0       # Initial exploration rate
    epsilon_end: float = 0.01        # Final exploration rate
    epsilon_decay_steps: int = 100_000
    batch_size: int = 64
    replay_buffer_size: int = 100_000
    target_sync_interval: int = 1_000  # Episodes between target net updates
    frame_stack: int = 3             # Number of stacked frames


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""
    num_episodes: int = 500_000
    checkpoint_interval: int = 10_000
    log_interval: int = 100
    eval_interval: int = 5_000
    eval_episodes: int = 100
    opponent_swap_interval: int = 50_000  # For self-play variant
    device: str = "cpu"              # "cuda" if GPU available


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    game: GameConfig = field(default_factory=GameConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
```

### Step 0.5 — Reproducibility Utility

Create `utils.py`:

```python
"""
utils.py — Utilities for reproducibility and common operations.
"""

import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set seeds for all random number generators.
    
    WHY THIS MATTERS:
        Neural network training involves randomness at many levels:
        - Weight initialization
        - Replay buffer sampling
        - Epsilon-greedy action selection
        - Environment resets
        
        Fixing all seeds makes experiments reproducible:
        same seed → same results, every time.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # These ensure deterministic behavior on GPU (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

---

## Phase 1 — The Game Engine

> **What we're building:** A pure-logic game engine with no rendering. It manages the grid, entities, collisions, and rewards. It must be fast — we'll run millions of episodes during training.

### Step 1.1 — The Grid

Create `game/grid.py`:

```python
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
```

### Step 1.2 — Actions and Directions

Create `game/actions.py`:

```python
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
```

### Step 1.3 — The Snake Entity

Create `game/snake.py`:

```python
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
```

### Step 1.4 — The Bait Entity

Create `game/bait.py`:

```python
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
```

### Step 1.5 — The Game Engine

Create `game/engine.py`:

```python
"""
game/engine.py — The core game loop and reward computation.

This is the heart of NeuroEvasion. The engine:
1. Manages the grid, snake, and bait
2. Processes actions from both agents each step
3. Computes rewards based on the outcome
4. Detects terminal conditions (capture, death, timeout)

SEPARATION OF CONCERNS:
    The engine knows NOTHING about neural networks, training, or rendering.
    It is pure game logic. This makes it testable, fast, and reusable.
"""

import random
import numpy as np
from game.grid import Grid, CellType
from game.snake import Snake
from game.bait import Bait
from game.actions import Action, DIRECTION_VECTORS
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
    
    def step(self, snake_action: int, bait_action: int) -> tuple[float, float, bool, dict]:
        """
        Execute one game step.
        
        Order of operations (critical for fairness):
            1. Snake chooses and moves
            2. Check if snake died (wall/self collision)
            3. Bait chooses and moves  
            4. Check if bait was captured
            5. Compute rewards
        
        Args:
            snake_action: integer action for the snake (0-3)
            bait_action: integer action for the bait (0-3, or 0-4 if STAY enabled)
            
        Returns:
            (snake_reward, bait_reward, done, info)
        """
        if self.done:
            raise RuntimeError("Game is over. Call reset() to start a new episode.")
        
        self.current_step += 1
        info = {"event": "step"}
        
        # Calculate distance BEFORE moves (for reward shaping)
        dist_before = self._manhattan_distance()
        
        # --- 1. Snake moves ---
        self.snake.set_direction(Action(snake_action))
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
            self.bait.move(Action(bait_action), self.grid)
        
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
        
        # --- 7. Compute shaping rewards ---
        dist_after = self._manhattan_distance()
        
        if dist_after < dist_before:
            # Snake got closer → reward snake, penalize bait
            snake_reward = self.rewards.distance_reward + self.rewards.step_penalty_snake
            bait_reward = -self.rewards.distance_reward + self.rewards.step_reward_bait
        elif dist_after > dist_before:
            # Bait escaped further → reward bait, penalize snake
            snake_reward = -self.rewards.distance_reward + self.rewards.step_penalty_snake
            bait_reward = self.rewards.distance_reward + self.rewards.step_reward_bait
        else:
            snake_reward = self.rewards.step_penalty_snake
            bait_reward = self.rewards.step_reward_bait
        
        info["distance"] = dist_after
        info["step"] = self.current_step
        
        self._update_grid()
        return (snake_reward, bait_reward, False, info)
    
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
            "grid": self.grid.cells.copy(),
            "snake_head": self.snake.head if self.snake else None,
            "snake_body": list(self.snake.body) if self.snake else [],
            "bait_pos": self.bait.position if self.bait else None,
            "step": self.current_step,
            "done": self.done,
            "winner": self.winner,
        }
```

> **📝 Teaching Note:** Notice how the engine is completely decoupled from rendering and AI. It takes integer actions and returns numerical rewards. This clean interface is the foundation of the Gymnasium API that the RL community standardized on.

---

## Phase 2 — Observation & Environment

> **What we're building:** A wrapper that converts the game engine's raw state into tensor observations suitable for neural networks, following the standard RL environment interface.

### Step 2.1 — State Encoder

Create `environment/state_encoder.py`:

```python
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
```

### Step 2.2 — Frame Stacker

Create `environment/frame_stacker.py`:

```python
"""
environment/frame_stacker.py — Stacks multiple observation frames.

WHY STACK FRAMES?
    A single frame is a snapshot — it shows WHERE things are but not
    which DIRECTION they're moving. By stacking the last k frames,
    the network can infer velocity and trajectory.
    
    This is the same technique DeepMind used for Atari games.
    Instead of adding recurrence (LSTM), we give the CNN a short
    "memory" through stacked frames.

    Example with k=3 and 4 channels per frame:
    Frame t-2: [body, head, bait, walls]  ← oldest
    Frame t-1: [body, head, bait, walls]
    Frame t:   [body, head, bait, walls]  ← newest
    
    Stacked: 12 channels × 20 × 20
"""

import numpy as np
from collections import deque


class FrameStacker:
    """
    Maintains a rolling window of the last k observation frames.
    
    On reset, fills the stack with copies of the initial observation.
    On each step, pushes the new frame and drops the oldest.
    """
    
    def __init__(self, num_frames: int = 3, obs_shape: tuple = (4, 20, 20)):
        self.num_frames = num_frames
        self.obs_shape = obs_shape
        self.frames: deque[np.ndarray] = deque(maxlen=num_frames)
    
    def reset(self, initial_obs: np.ndarray) -> np.ndarray:
        """Fill the stack with copies of the initial observation."""
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(initial_obs.copy())
        return self.get_stacked()
    
    def push(self, obs: np.ndarray) -> np.ndarray:
        """Add a new frame and return the stacked observation."""
        self.frames.append(obs.copy())
        return self.get_stacked()
    
    def get_stacked(self) -> np.ndarray:
        """
        Concatenate all frames along the channel dimension.
        
        Returns:
            np.ndarray of shape (num_frames * channels, H, W)
        """
        return np.concatenate(list(self.frames), axis=0)
```

### Step 2.3 — Environment Wrapper

Create `environment/env.py`:

```python
"""
environment/env.py — Gymnasium-compatible environment wrapper.

This wraps the GameEngine + StateEncoder + FrameStacker into a clean
interface that the training loop and agents interact with.

THE GYM API PATTERN:
    env.reset()  → observations
    env.step(actions) → observations, rewards, done, info
    
    This is the standard interface used across the RL community
    (OpenAI Gym, Gymnasium, DeepMind Lab, etc.). Learning this
    pattern transfers to any RL project.
"""

import numpy as np
from game.engine import GameEngine
from environment.state_encoder import StateEncoder
from environment.frame_stacker import FrameStacker
from config import Config


class NeuroEvasionEnv:
    """
    Multi-agent environment for the NeuroEvasion game.
    
    Provides separate observations for snake and bait agents.
    
    Usage:
        env = NeuroEvasionEnv(config)
        snake_obs, bait_obs = env.reset()
        
        while not done:
            snake_obs, bait_obs, snake_r, bait_r, done, info = env.step(s_act, b_act)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.engine = GameEngine(config.game, config.rewards)
        self.encoder = StateEncoder(config.game.grid_size)
        
        obs_shape = (4, config.game.grid_size, config.game.grid_size)
        self.snake_stacker = FrameStacker(config.agent.frame_stack, obs_shape)
        self.bait_stacker = FrameStacker(config.agent.frame_stack, obs_shape)
        
        # Action and observation space info
        self.snake_num_actions = 4   # UP, DOWN, LEFT, RIGHT
        self.bait_num_actions = 5 if config.game.bait_can_stay else 4
        self.obs_channels = 4 * config.agent.frame_stack
        self.obs_height = config.game.grid_size
        self.obs_width = config.game.grid_size
    
    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Reset the environment for a new episode.
        
        Returns:
            (snake_observation, bait_observation) — stacked frame tensors
        """
        info = self.engine.reset()
        state = self.engine.get_state()
        
        snake_obs = self.encoder.encode_for_snake(
            self.engine.grid, state["snake_body"],
            state["snake_head"], state["bait_pos"]
        )
        bait_obs = self.encoder.encode_for_bait(
            self.engine.grid, state["snake_body"],
            state["snake_head"], state["bait_pos"]
        )
        
        snake_stacked = self.snake_stacker.reset(snake_obs)
        bait_stacked = self.bait_stacker.reset(bait_obs)
        
        return snake_stacked, bait_stacked
    
    def step(self, snake_action: int, bait_action: int) -> tuple:
        """
        Execute one environment step.
        
        Returns:
            (snake_obs, bait_obs, snake_reward, bait_reward, done, info)
        """
        snake_reward, bait_reward, done, info = self.engine.step(snake_action, bait_action)
        
        state = self.engine.get_state()
        
        if not done:
            snake_obs = self.encoder.encode_for_snake(
                self.engine.grid, state["snake_body"],
                state["snake_head"], state["bait_pos"]
            )
            bait_obs = self.encoder.encode_for_bait(
                self.engine.grid, state["snake_body"],
                state["snake_head"], state["bait_pos"]
            )
        else:
            # Terminal state — return zeros (won't be used for learning)
            snake_obs = np.zeros_like(self.snake_stacker.get_stacked()[:4])
            bait_obs = np.zeros_like(self.bait_stacker.get_stacked()[:4])
        
        snake_stacked = self.snake_stacker.push(snake_obs)
        bait_stacked = self.bait_stacker.push(bait_obs)
        
        return snake_stacked, bait_stacked, snake_reward, bait_reward, done, info
```

---

## Phase 3 — Neural Network Agents

> **What we're building:** The brains of both agents — a Deep Q-Network (DQN) that learns to map observations to optimal actions, supported by experience replay and epsilon-greedy exploration.

### Step 3.1 — The DQN Architecture

Create `agents/networks.py`:

```python
"""
agents/networks.py — Neural network architectures for DQN agents.

ARCHITECTURE OVERVIEW:
    We use a Convolutional Neural Network (CNN) because our observations
    are 2D grids (like images). The CNN learns spatial patterns:
    
    Conv layers → extract features ("snake is nearby", "wall ahead")
    FC layers   → combine features into action-value estimates
    
WHAT IS A Q-VALUE?
    Q(s, a) estimates the total future reward if we take action 'a'
    in state 's' and then follow our policy. The agent picks the
    action with the highest Q-value: argmax_a Q(s, a).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Deep Q-Network with convolutional feature extraction.
    
    Architecture:
        3 × Conv2d layers (feature extraction)
        → Flatten
        → 2 × Linear layers (decision making)
        → Q-value output (one per action)
    
    Args:
        in_channels: number of input channels (e.g., 12 for 3-frame stack × 4 channels)
        grid_size: height/width of the grid (e.g., 20)
        num_actions: number of possible actions (e.g., 4 or 5)
    """
    
    def __init__(self, in_channels: int, grid_size: int, num_actions: int):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Layer 1: Detect basic patterns (edges, positions)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Layer 2: Combine basic patterns into complex features
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 3: Higher-level strategic features
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Calculate flattened size after convolutions
        # With padding=1 and stride=1, spatial dimensions are preserved
        flat_size = 64 * grid_size * grid_size
        
        self.fc_layers = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),  # Output: Q(s,a) for each action
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: observation → Q-values.
        
        Args:
            x: observation tensor of shape (batch, channels, H, W)
            
        Returns:
            Q-values of shape (batch, num_actions)
        """
        features = self.conv_layers(x)
        flat = features.view(features.size(0), -1)  # Flatten
        q_values = self.fc_layers(flat)
        return q_values


class DuelingDQN(nn.Module):
    """
    Dueling DQN — separates state value from action advantage.
    
    KEY INSIGHT:
        Sometimes it's important to know the VALUE of a state regardless
        of which action we take. For example, if we're about to die,
        ALL actions have low value. The Dueling architecture learns this
        decomposition explicitly:
        
        Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))
        
        V(s)    = "how good is this state in general?"
        A(s, a) = "how much better/worse is this specific action?"
        
    PAPER: Wang et al., "Dueling Network Architectures for Deep RL" (2016)
    """
    
    def __init__(self, in_channels: int, grid_size: int, num_actions: int):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        flat_size = 64 * grid_size * grid_size
        
        # Value stream: "How good is this state?"
        self.value_stream = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        
        # Advantage stream: "How much better is each action?"
        self.advantage_stream = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        flat = features.view(features.size(0), -1)
        
        value = self.value_stream(flat)           # (batch, 1)
        advantage = self.advantage_stream(flat)   # (batch, actions)
        
        # Q = V + (A - mean(A))  → ensures identifiability
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
```

### Step 3.2 — Experience Replay Buffer

Create `agents/replay_buffer.py`:

```python
"""
agents/replay_buffer.py — Experience replay memory.

WHY REPLAY BUFFERS?
    Neural networks assume training data is independent and identically
    distributed (i.i.d.). But consecutive game frames are heavily correlated:
    frame t and frame t+1 look almost identical.
    
    The replay buffer stores past experiences and lets us sample RANDOM
    mini-batches, breaking temporal correlations and providing i.i.d.-like
    training data.
    
    Think of it as the agent's "memory" — it remembers past experiences
    and learns from random flashbacks instead of just the most recent event.

ANALOGY FOR STUDENTS:
    Imagine studying for an exam by only re-reading the last paragraph.
    That's what training without replay is. With replay, you randomly
    flip to different pages — much better learning!
"""

import random
import numpy as np
from collections import deque, namedtuple

# A single experience tuple
Transition = namedtuple('Transition', 
    ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """
    Fixed-size circular buffer that stores experience transitions.
    
    When full, oldest experiences are overwritten (FIFO).
    Sampling is uniform random.
    
    Args:
        capacity: maximum number of transitions to store
    """
    
    def __init__(self, capacity: int = 100_000):
        self.buffer: deque[Transition] = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Store a transition in memory."""
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> dict:
        """
        Sample a random batch of transitions.
        
        Returns a dict of numpy arrays, ready for conversion to tensors:
            states:      (batch, channels, H, W)
            actions:     (batch,)
            rewards:     (batch,)
            next_states: (batch, channels, H, W)
            dones:       (batch,)  — boolean mask
        """
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        
        return {
            'states': np.array(batch.state),
            'actions': np.array(batch.action),
            'rewards': np.array(batch.reward, dtype=np.float32),
            'next_states': np.array(batch.next_state),
            'dones': np.array(batch.done, dtype=np.float32),
        }
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if we have enough samples for a batch."""
        return len(self) >= batch_size
```

### Step 3.3 — The DQN Agent

Create `agents/dqn_agent.py`:

```python
"""
agents/dqn_agent.py — The DQN agent that learns to play.

This class ties everything together:
    - Uses the DQN network to estimate Q-values
    - Selects actions via ε-greedy exploration
    - Stores experiences in the replay buffer
    - Trains the network using TD-learning
    - Maintains a target network for stable training

THE TRAINING ALGORITHM (pseudocode):
    1. Observe state s
    2. With probability ε: random action, else: argmax Q(s, a)
    3. Execute action, observe reward r and next state s'
    4. Store (s, a, r, s', done) in replay buffer
    5. Sample random batch from buffer
    6. Compute target: y = r + γ × max Q_target(s', a')
    7. Compute loss: L = (Q(s, a) - y)²
    8. Backpropagate and update weights
    9. Periodically copy weights to target network
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from agents.networks import DQNNetwork, DuelingDQN
from agents.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    
    Args:
        in_channels: observation channels (frame_stack × 4)
        grid_size: grid height/width
        num_actions: number of available actions
        config: AgentConfig dataclass
        device: 'cpu' or 'cuda'
        use_dueling: if True, use Dueling DQN architecture
    """
    
    def __init__(self, in_channels: int, grid_size: int, num_actions: int,
                 config, device: str = "cpu", use_dueling: bool = False):
        self.config = config
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.steps_done = 0
        
        # --- Networks ---
        NetworkClass = DuelingDQN if use_dueling else DQNNetwork
        
        self.policy_net = NetworkClass(in_channels, grid_size, num_actions).to(self.device)
        self.target_net = NetworkClass(in_channels, grid_size, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is never trained directly
        
        # --- Optimizer ---
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        
        # --- Replay Buffer ---
        self.memory = ReplayBuffer(capacity=config.replay_buffer_size)
        
        # --- Exploration ---
        self.epsilon = config.epsilon_start
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay_steps = config.epsilon_decay_steps
    
    def select_action(self, state: np.ndarray) -> int:
        """
        ε-greedy action selection.
        
        With probability ε: explore (random action)
        With probability 1-ε: exploit (best Q-value action)
        
        WHY ε-GREEDY?
            Early in training, the network's Q-values are random noise.
            We need exploration to discover which actions lead to rewards.
            As training progresses, we trust the network more (lower ε).
        """
        self.steps_done += 1
        self._update_epsilon()
        
        if np.random.random() < self.epsilon:
            # Explore: choose a random action
            return np.random.randint(self.num_actions)
        else:
            # Exploit: choose the action with highest Q-value
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def _update_epsilon(self) -> None:
        """Linearly decay ε from start to end over decay_steps."""
        progress = min(1.0, self.steps_done / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
    
    def store_transition(self, state, action, reward, next_state, done) -> None:
        """Save an experience to replay memory."""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float | None:
        """
        Perform one training step (sample batch + gradient update).
        
        THE CORE DQN UPDATE:
            target = reward + γ × max_a' Q_target(next_state, a')
            loss = MSE(Q_policy(state, action), target)
            
        Returns:
            loss value (float) or None if buffer isn't ready
        """
        if not self.memory.is_ready(self.config.batch_size):
            return None
        
        batch = self.memory.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        # --- Current Q-values ---
        # Q(s, a) for the actions we actually took
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # --- Target Q-values ---
        with torch.no_grad():
            # max_a' Q_target(s', a')
            next_q = self.target_net(next_states).max(dim=1)[0]
            # If episode is done, there is no next state value
            target_q = rewards + self.config.gamma * next_q * (1 - dones)
        
        # --- Compute loss and backprop ---
        loss = nn.functional.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        return loss.item()
    
    def sync_target_network(self) -> None:
        """
        Copy policy network weights to target network.
        
        WHY A SEPARATE TARGET NETWORK?
            If we used the same network for both Q(s,a) and the target,
            we'd be chasing a moving target — the network updates change
            the target values, creating instability.
            
            The target network provides a STABLE reference point.
            We update it periodically (every ~1000 episodes).
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
        }, path)
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state (useful for visualization/debugging)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_tensor).cpu().numpy()[0]
```

> **📝 Teaching Note — The DQN Training Loop Explained:**
>
> ```
> Step 1: Agent sees state s = [grid with snake, bait, walls]
> Step 2: ε = 0.7 → 70% chance of random action
>         Agent randomly picks RIGHT
> Step 3: Environment returns: reward = +0.1 (got closer to bait)
>         New state s' = [updated grid]
> Step 4: Store (s, RIGHT, +0.1, s', False) in replay buffer
> Step 5: Sample 64 random experiences from buffer
> Step 6: For each experience, compute target:
>         target = 0.1 + 0.99 × max(Q_target(s'))
>         target = 0.1 + 0.99 × 3.2 = 3.268
> Step 7: Current Q(s, RIGHT) = 2.5
>         Loss = (2.5 - 3.268)² = 0.59
> Step 8: Backprop → update weights to make Q(s, RIGHT) closer to 3.268
> ```
>
> Over thousands of iterations, the Q-values converge to the true expected returns, and the agent learns the optimal policy.

---

## Phase 4 — Training & Co-Evolution

> **What we're building:** The training orchestrator that runs both agents against each other for hundreds of thousands of episodes, logging metrics and saving checkpoints.

### Step 4.1 — Metrics Logger

Create `training/logger.py`:

```python
"""
training/logger.py — Training metrics logging.

Logs to both TensorBoard (interactive graphs) and CSV (raw data).
TensorBoard lets you monitor training in real-time with:
    tensorboard --logdir logs/
"""

import os
import csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """
    Dual logger: TensorBoard + CSV file.
    
    Usage:
        logger = TrainingLogger("logs/run_001")
        logger.log_episode(episode=100, snake_reward=5.2, ...)
        logger.close()
    """
    
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        csv_path = os.path.join(log_dir, "metrics.csv")
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "episode", "snake_reward", "bait_reward", "snake_loss",
            "bait_loss", "winner", "steps", "epsilon", "timestamp"
        ])
    
    def log_episode(self, episode: int, snake_reward: float, bait_reward: float,
                    snake_loss: float, bait_loss: float, winner: str,
                    steps: int, epsilon: float) -> None:
        """Log metrics for a completed episode."""
        # TensorBoard
        self.writer.add_scalar("Reward/Snake", snake_reward, episode)
        self.writer.add_scalar("Reward/Bait", bait_reward, episode)
        self.writer.add_scalar("Loss/Snake", snake_loss, episode)
        self.writer.add_scalar("Loss/Bait", bait_loss, episode)
        self.writer.add_scalar("Episode/Steps", steps, episode)
        self.writer.add_scalar("Episode/Epsilon", epsilon, episode)
        
        snake_win = 1.0 if winner == "snake" else 0.0
        self.writer.add_scalar("WinRate/Snake", snake_win, episode)
        
        # CSV
        self.csv_writer.writerow([
            episode, f"{snake_reward:.4f}", f"{bait_reward:.4f}",
            f"{snake_loss:.6f}", f"{bait_loss:.6f}",
            winner, steps, f"{epsilon:.4f}", datetime.now().isoformat()
        ])
    
    def log_evaluation(self, episode: int, snake_win_rate: float,
                       avg_survival: float) -> None:
        """Log evaluation round metrics."""
        self.writer.add_scalar("Eval/SnakeWinRate", snake_win_rate, episode)
        self.writer.add_scalar("Eval/AvgSurvivalSteps", avg_survival, episode)
    
    def close(self) -> None:
        self.writer.close()
        self.csv_file.close()
```

### Step 4.2 — The Training Orchestrator

Create `training/trainer.py`:

```python
"""
training/trainer.py — Main training loop for co-evolutionary learning.

THE CO-EVOLUTION PROCESS:
    1. Both agents start with random policies (high ε)
    2. They play thousands of games against each other
    3. Each agent improves against the CURRENT version of the other
    4. As one gets better, it forces the other to adapt
    5. This creates an arms race of increasingly sophisticated strategies

EMERGENT BEHAVIORS YOU MAY OBSERVE:
    - Snake learns to "corner" the bait against walls
    - Bait learns to run along walls to maximize escape routes
    - Snake learns to "cut off" the bait's escape paths
    - Bait learns to fake one direction then go another
"""

import os
import time
import numpy as np
from config import Config
from environment.env import NeuroEvasionEnv
from agents.dqn_agent import DQNAgent
from training.logger import TrainingLogger
from utils import set_global_seed


def train(config: Config) -> None:
    """
    Main training function.
    
    Creates the environment and both agents, then runs the
    co-evolutionary training loop.
    """
    # --- Setup ---
    set_global_seed(config.seed)
    
    env = NeuroEvasionEnv(config)
    
    in_channels = env.obs_channels
    grid_size = config.game.grid_size
    
    snake_agent = DQNAgent(
        in_channels=in_channels,
        grid_size=grid_size,
        num_actions=env.snake_num_actions,
        config=config.agent,
        device=config.training.device,
    )
    
    bait_agent = DQNAgent(
        in_channels=in_channels,
        grid_size=grid_size,
        num_actions=env.bait_num_actions,
        config=config.agent,
        device=config.training.device,
    )
    
    logger = TrainingLogger(f"logs/run_{int(time.time())}")
    
    print(f"Starting training for {config.training.num_episodes} episodes...")
    print(f"Grid: {grid_size}×{grid_size} | Device: {config.training.device}")
    print(f"Snake actions: {env.snake_num_actions} | Bait actions: {env.bait_num_actions}")
    print("-" * 60)
    
    # --- Tracking ---
    episode_rewards_snake = []
    episode_rewards_bait = []
    snake_wins = 0
    total_games = 0
    
    # --- Training Loop ---
    for episode in range(1, config.training.num_episodes + 1):
        snake_obs, bait_obs = env.reset()
        
        ep_snake_reward = 0.0
        ep_bait_reward = 0.0
        ep_snake_loss = 0.0
        ep_bait_loss = 0.0
        loss_count = 0
        
        for step in range(config.game.max_steps):
            # Both agents select actions
            snake_action = snake_agent.select_action(snake_obs)
            bait_action = bait_agent.select_action(bait_obs)
            
            # Environment step
            (snake_obs_next, bait_obs_next,
             snake_reward, bait_reward, done, info) = env.step(snake_action, bait_action)
            
            # Store experiences
            snake_agent.store_transition(
                snake_obs, snake_action, snake_reward, snake_obs_next, done)
            bait_agent.store_transition(
                bait_obs, bait_action, bait_reward, bait_obs_next, done)
            
            # Train both agents
            s_loss = snake_agent.train_step()
            b_loss = bait_agent.train_step()
            
            if s_loss is not None:
                ep_snake_loss += s_loss
                ep_bait_loss += b_loss if b_loss else 0
                loss_count += 1
            
            ep_snake_reward += snake_reward
            ep_bait_reward += bait_reward
            
            snake_obs = snake_obs_next
            bait_obs = bait_obs_next
            
            if done:
                break
        
        # --- Post-Episode ---
        total_games += 1
        if info.get("event") == "capture":
            snake_wins += 1
        
        avg_s_loss = ep_snake_loss / max(loss_count, 1)
        avg_b_loss = ep_bait_loss / max(loss_count, 1)
        
        # Sync target networks periodically
        if episode % config.agent.target_sync_interval == 0:
            snake_agent.sync_target_network()
            bait_agent.sync_target_network()
        
        # Log metrics
        logger.log_episode(
            episode=episode,
            snake_reward=ep_snake_reward,
            bait_reward=ep_bait_reward,
            snake_loss=avg_s_loss,
            bait_loss=avg_b_loss,
            winner=info.get("event", "unknown"),
            steps=step + 1,
            epsilon=snake_agent.epsilon,
        )
        
        # Print progress
        if episode % config.training.log_interval == 0:
            win_rate = snake_wins / total_games * 100
            print(
                f"Ep {episode:>7d} | "
                f"ε={snake_agent.epsilon:.3f} | "
                f"Snake R={ep_snake_reward:>7.2f} | "
                f"Bait R={ep_bait_reward:>7.2f} | "
                f"Win%={win_rate:.1f}% | "
                f"Steps={step+1:>3d}"
            )
            # Reset rolling counters
            snake_wins = 0
            total_games = 0
        
        # Save checkpoints
        if episode % config.training.checkpoint_interval == 0:
            snake_agent.save(f"checkpoints/snake_ep{episode}.pt")
            bait_agent.save(f"checkpoints/bait_ep{episode}.pt")
            print(f"  💾 Checkpoints saved at episode {episode}")
    
    # --- Final Save ---
    snake_agent.save("checkpoints/snake_final.pt")
    bait_agent.save("checkpoints/bait_final.pt")
    logger.close()
    print("\n✅ Training complete!")
```

### Step 4.3 — CLI Entry Point

Create `main.py`:

```python
"""
main.py — Command-line entry point for NeuroEvasion.

Usage:
    python main.py train                    # Train agents
    python main.py train --episodes 10000   # Custom episode count
    python main.py demo                     # Watch trained agents play
    python main.py evaluate                 # Evaluate model performance
"""

import argparse
import sys
from config import Config


def main():
    parser = argparse.ArgumentParser(
        description="NeuroEvasion — Co-Evolutionary Pursuit-Evasion with DNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # --- Train command ---
    train_parser = subparsers.add_parser("train", help="Train the agents")
    train_parser.add_argument("--episodes", type=int, default=None,
                              help="Number of training episodes")
    train_parser.add_argument("--grid-size", type=int, default=None,
                              help="Grid size (N×N)")
    train_parser.add_argument("--lr", type=float, default=None,
                              help="Learning rate")
    train_parser.add_argument("--device", type=str, default=None,
                              choices=["cpu", "cuda"],
                              help="Training device")
    train_parser.add_argument("--seed", type=int, default=None,
                              help="Random seed")
    
    # --- Demo command ---
    demo_parser = subparsers.add_parser("demo", help="Watch agents play")
    demo_parser.add_argument("--snake-model", type=str, required=True,
                             help="Path to snake checkpoint")
    demo_parser.add_argument("--bait-model", type=str, required=True,
                             help="Path to bait checkpoint")
    demo_parser.add_argument("--speed", type=int, default=10,
                             help="Game speed (FPS)")
    
    # --- Evaluate command ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate agents")
    eval_parser.add_argument("--snake-model", type=str, required=True)
    eval_parser.add_argument("--bait-model", type=str, required=True)
    eval_parser.add_argument("--num-games", type=int, default=1000)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    config = Config()
    
    if args.command == "train":
        if args.episodes:
            config.training.num_episodes = args.episodes
        if args.grid_size:
            config.game.grid_size = args.grid_size
        if args.lr:
            config.agent.learning_rate = args.lr
        if args.device:
            config.training.device = args.device
        if args.seed:
            config.seed = args.seed
        
        from training.trainer import train
        train(config)
    
    elif args.command == "demo":
        from visualization.renderer import run_demo
        run_demo(config, args.snake_model, args.bait_model, args.speed)
    
    elif args.command == "evaluate":
        from evaluation.evaluator import evaluate
        evaluate(config, args.snake_model, args.bait_model, args.num_games)


if __name__ == "__main__":
    main()
```

---

## Phase 5 — Visualization & Evaluation

> **What we're building:** A Pygame renderer to watch trained agents play in real-time, an evaluator to measure agent performance, and plotting utilities for training analysis.

### Step 5.1 — Pygame Renderer

Create `visualization/renderer.py`:

```python
"""
visualization/renderer.py — Real-time game visualization with Pygame.

This renders the game grid with color-coded entities and a HUD
showing episode stats. Used in 'demo' mode to watch trained agents.

DESIGN:
    - Dark background with neon-style colors for a modern look
    - Snake rendered as a gradient from bright to dark green
    - Bait pulses with a glow effect
    - HUD shows real-time Q-values and scores
"""

import pygame
import numpy as np
import sys
import math
from config import Config
from environment.env import NeuroEvasionEnv
from agents.dqn_agent import DQNAgent

# --- Color Palette ---
BLACK = (10, 10, 15)
DARK_GRAY = (25, 25, 35)
WALL_COLOR = (40, 45, 60)
GRID_LINE = (30, 30, 40)
SNAKE_HEAD = (0, 255, 120)
SNAKE_BODY = (0, 180, 80)
SNAKE_TAIL = (0, 100, 50)
BAIT_COLOR = (255, 60, 80)
BAIT_GLOW = (255, 100, 120)
TEXT_COLOR = (200, 210, 230)
HIGHLIGHT = (100, 200, 255)


class GameRenderer:
    """Renders the NeuroEvasion game with Pygame."""
    
    def __init__(self, grid_size: int, cell_size: int = 30):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.hud_width = 280
        self.width = grid_size * cell_size + self.hud_width
        self.height = grid_size * cell_size
        
        pygame.init()
        pygame.display.set_caption("🧠 NeuroEvasion — Pursuit-Evasion AI")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 22, bold=True)
        self.frame_count = 0
    
    def render(self, state: dict, info: dict = None) -> None:
        """Draw the current game state."""
        self.screen.fill(BLACK)
        self.frame_count += 1
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            x = i * self.cell_size
            pygame.draw.line(self.screen, GRID_LINE, (x, 0), (x, self.height), 1)
            pygame.draw.line(self.screen, GRID_LINE, (0, x), (self.grid_size * self.cell_size, x), 1)
        
        grid = state["grid"]
        snake_body = state.get("snake_body", [])
        bait_pos = state.get("bait_pos")
        
        # Draw walls
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r, c] == 1:  # WALL
                    rect = pygame.Rect(c * self.cell_size, r * self.cell_size,
                                       self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, WALL_COLOR, rect)
        
        # Draw snake body with gradient
        if snake_body:
            for i, (r, c) in enumerate(snake_body):
                ratio = i / max(len(snake_body) - 1, 1)
                color = self._lerp_color(SNAKE_HEAD, SNAKE_TAIL, ratio)
                rect = pygame.Rect(c * self.cell_size + 1, r * self.cell_size + 1,
                                   self.cell_size - 2, self.cell_size - 2)
                pygame.draw.rect(self.screen, color, rect, border_radius=4)
            
            # Snake head highlight
            hr, hc = snake_body[0]
            rect = pygame.Rect(hc * self.cell_size + 1, hr * self.cell_size + 1,
                               self.cell_size - 2, self.cell_size - 2)
            pygame.draw.rect(self.screen, SNAKE_HEAD, rect, border_radius=6)
        
        # Draw bait with pulse effect
        if bait_pos:
            br, bc = bait_pos
            pulse = abs(math.sin(self.frame_count * 0.1)) * 4
            cx = bc * self.cell_size + self.cell_size // 2
            cy = br * self.cell_size + self.cell_size // 2
            radius = self.cell_size // 2 - 2 + int(pulse)
            
            # Glow effect
            glow_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*BAIT_GLOW, 40), (radius * 2, radius * 2), radius * 2)
            self.screen.blit(glow_surf, (cx - radius * 2, cy - radius * 2))
            
            # Core
            pygame.draw.circle(self.screen, BAIT_COLOR, (cx, cy), radius - 2)
        
        # Draw HUD
        self._draw_hud(state, info)
        
        pygame.display.flip()
    
    def _draw_hud(self, state: dict, info: dict) -> None:
        """Draw the heads-up display panel."""
        hud_x = self.grid_size * self.cell_size + 10
        
        # Panel background
        panel_rect = pygame.Rect(self.grid_size * self.cell_size, 0,
                                  self.hud_width, self.height)
        pygame.draw.rect(self.screen, DARK_GRAY, panel_rect)
        
        # Title
        title = self.font_large.render("NeuroEvasion", True, HIGHLIGHT)
        self.screen.blit(title, (hud_x, 15))
        
        y = 55
        lines = [
            f"Step: {state.get('step', 0)}",
            f"Done: {state.get('done', False)}",
            f"Winner: {state.get('winner', '-')}",
            "",
            f"Snake len: {len(state.get('snake_body', []))}",
        ]
        
        if info:
            lines.extend([
                f"Distance: {info.get('distance', '?')}",
                f"Event: {info.get('event', 'step')}",
            ])
        
        for line in lines:
            if line:
                surf = self.font.render(line, True, TEXT_COLOR)
                self.screen.blit(surf, (hud_x, y))
            y += 22
    
    def _lerp_color(self, c1, c2, t):
        """Linear interpolation between two colors."""
        return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))
    
    def handle_events(self) -> str:
        """Handle Pygame events. Returns 'quit', 'pause', or 'continue'."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key == pygame.K_p:
                    return "pause"
                if event.key == pygame.K_r:
                    return "restart"
        return "continue"
    
    def close(self) -> None:
        pygame.quit()


def run_demo(config: Config, snake_model_path: str, 
             bait_model_path: str, speed: int = 10) -> None:
    """Run a visual demo with trained agents."""
    env = NeuroEvasionEnv(config)
    renderer = GameRenderer(config.game.grid_size)
    
    in_channels = env.obs_channels
    grid_size = config.game.grid_size
    
    snake_agent = DQNAgent(in_channels, grid_size, env.snake_num_actions,
                           config.agent, device="cpu")
    bait_agent = DQNAgent(in_channels, grid_size, env.bait_num_actions,
                          config.agent, device="cpu")
    
    snake_agent.load(snake_model_path)
    bait_agent.load(bait_model_path)
    
    # Set to pure exploitation (no random actions)
    snake_agent.epsilon = 0.0
    bait_agent.epsilon = 0.0
    
    print("🎮 Demo mode — Press Q to quit, P to pause, R to restart")
    
    while True:
        snake_obs, bait_obs = env.reset()
        done = False
        
        while not done:
            event = renderer.handle_events()
            if event == "quit":
                renderer.close()
                return
            if event == "restart":
                break
            if event == "pause":
                while renderer.handle_events() != "pause":
                    renderer.clock.tick(10)
            
            snake_action = snake_agent.select_action(snake_obs)
            bait_action = bait_agent.select_action(bait_obs)
            
            snake_obs, bait_obs, _, _, done, info = env.step(snake_action, bait_action)
            
            state = env.engine.get_state()
            renderer.render(state, info)
            renderer.clock.tick(speed)
    
    renderer.close()
```

### Step 5.2 — Evaluator

Create `evaluation/evaluator.py`:

```python
"""
evaluation/evaluator.py — Evaluate trained agents over many games.

Computes key performance metrics without rendering, for fast evaluation.
"""

import numpy as np
from config import Config
from environment.env import NeuroEvasionEnv
from agents.dqn_agent import DQNAgent


def evaluate(config: Config, snake_model_path: str, 
             bait_model_path: str, num_games: int = 1000) -> dict:
    """
    Run evaluation games and compute statistics.
    
    Returns:
        dict with snake_win_rate, avg_survival, avg_snake_reward, etc.
    """
    env = NeuroEvasionEnv(config)
    in_channels = env.obs_channels
    grid_size = config.game.grid_size
    
    snake_agent = DQNAgent(in_channels, grid_size, env.snake_num_actions,
                           config.agent, device="cpu")
    bait_agent = DQNAgent(in_channels, grid_size, env.bait_num_actions,
                          config.agent, device="cpu")
    
    snake_agent.load(snake_model_path)
    bait_agent.load(bait_model_path)
    snake_agent.epsilon = 0.0
    bait_agent.epsilon = 0.0
    
    snake_wins = 0
    survival_steps = []
    snake_rewards = []
    bait_rewards = []
    
    for game in range(num_games):
        snake_obs, bait_obs = env.reset()
        ep_snake_r = 0.0
        ep_bait_r = 0.0
        done = False
        steps = 0
        
        while not done:
            s_action = snake_agent.select_action(snake_obs)
            b_action = bait_agent.select_action(bait_obs)
            snake_obs, bait_obs, s_r, b_r, done, info = env.step(s_action, b_action)
            ep_snake_r += s_r
            ep_bait_r += b_r
            steps += 1
        
        if info.get("event") == "capture":
            snake_wins += 1
        
        survival_steps.append(steps)
        snake_rewards.append(ep_snake_r)
        bait_rewards.append(ep_bait_r)
        
        if (game + 1) % 100 == 0:
            print(f"  Evaluated {game + 1}/{num_games} games...")
    
    results = {
        "snake_win_rate": snake_wins / num_games,
        "bait_win_rate": 1 - snake_wins / num_games,
        "avg_survival_steps": np.mean(survival_steps),
        "median_survival_steps": np.median(survival_steps),
        "avg_snake_reward": np.mean(snake_rewards),
        "avg_bait_reward": np.mean(bait_rewards),
        "total_games": num_games,
    }
    
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Games played:          {num_games}")
    print(f"  Snake win rate:        {results['snake_win_rate']:.1%}")
    print(f"  Bait win rate:         {results['bait_win_rate']:.1%}")
    print(f"  Avg survival (steps):  {results['avg_survival_steps']:.1f}")
    print(f"  Avg snake reward:      {results['avg_snake_reward']:.2f}")
    print(f"  Avg bait reward:       {results['avg_bait_reward']:.2f}")
    print("=" * 50)
    
    return results
```

---

## Quick-Start Guide

### 1. Install & Setup

```bash
git clone https://github.com/YOUR_USERNAME/NeuroEvasion.git
cd NeuroEvasion
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train (Short Run)

```bash
# Quick smoke test: 1,000 episodes (~2 minutes)
python main.py train --episodes 1000

# Full training: 500,000 episodes (several hours)
python main.py train

# With GPU
python main.py train --device cuda
```

### 3. Monitor Training

```bash
# In a separate terminal:
tensorboard --logdir logs/
# Open http://localhost:6006 in your browser
```

### 4. Watch Demo

```bash
python main.py demo \
  --snake-model checkpoints/snake_final.pt \
  --bait-model checkpoints/bait_final.pt \
  --speed 8
```

### 5. Evaluate

```bash
python main.py evaluate \
  --snake-model checkpoints/snake_final.pt \
  --bait-model checkpoints/bait_final.pt \
  --num-games 1000
```

---

## Concepts Cheat Sheet for Students

| Concept | Where in Code | What It Does |
|---------|---------------|-------------|
| **CNN** | `agents/networks.py` | Extracts spatial features from grid observation |
| **Q-Learning** | `agents/dqn_agent.py` → `train_step()` | Updates Q-values toward TD targets |
| **Experience Replay** | `agents/replay_buffer.py` | Breaks temporal correlation in training data |
| **Target Network** | `agents/dqn_agent.py` → `sync_target_network()` | Stabilizes training with a frozen reference |
| **ε-Greedy** | `agents/dqn_agent.py` → `select_action()` | Balances exploration vs. exploitation |
| **Reward Shaping** | `game/engine.py` → `step()` | Guides learning with distance-based rewards |
| **Zero-Sum Game** | `config.py` → `RewardConfig` | Snake reward = −Bait reward |
| **Co-Evolution** | `training/trainer.py` | Both agents improve against each other |
| **Frame Stacking** | `environment/frame_stacker.py` | Gives the CNN a sense of motion/velocity |
| **Discount Factor (γ)** | `config.py` → `AgentConfig.gamma` | How much future rewards matter vs. immediate |

---

## Suggested Exercises for Students

1. **Modify the reward function** — What happens if you remove distance shaping and only keep terminal rewards? Does learning still converge?

2. **Change the grid size** — Try 10×10 (easier) and 30×30 (harder). How does this affect training time and strategy complexity?

3. **Give the bait a speed advantage** — Set `bait_move_every=1` and give bait 2 moves per snake move. Can the snake still learn to catch it?

4. **Implement Double DQN** — Use the policy network to SELECT the best next action, but the target network to EVALUATE it. Does this reduce Q-value overestimation?

5. **Add obstacles** — Place random walls inside the arena. How do the agents' strategies adapt?

6. **Visualize Q-values** — Render a heatmap of Q-values across the grid. Where does the snake think it's "winning"?

7. **Asymmetric information** — What if the bait can only see a 5×5 area around itself (limited vision)? How does this change the game dynamics?

---

## Recommended Reading

| Topic | Paper/Resource |
|-------|---------------|
| DQN | Mnih et al., "Playing Atari with Deep RL" (2013) |
| Target Networks | Mnih et al., "Human-level control through deep RL" (2015) |
| Dueling DQN | Wang et al., "Dueling Network Architectures" (2016) |
| Multi-Agent RL | Lowe et al., "Multi-Agent Actor-Critic" (2017) |
| Self-Play | Silver et al., "Mastering Go without Human Knowledge" (2017) |
| Reward Shaping | Ng et al., "Policy Invariance Under Reward Transformations" (1999) |

---

> **🎓 End of Tutorial.** You now have a complete understanding of every component in the NeuroEvasion system. Build each phase incrementally, test thoroughly, and watch your agents evolve!
