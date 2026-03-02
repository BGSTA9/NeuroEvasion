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

*Continued in Phase 4…*
]]>
