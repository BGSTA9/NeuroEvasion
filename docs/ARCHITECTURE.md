<![CDATA[# 🧠 NeuroEvasion — Architecture & Build Phases

> **Transforming the classic Snake & Bait game into a co-evolutionary, zero-sum pursuit-evasion simulation powered by Deep Neural Networks.**

---

## Table of Contents

1. [Project Vision](#1-project-vision)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Phase 0 — Foundation & Environment](#3-phase-0--foundation--environment)
4. [Phase 1 — The Game Engine](#4-phase-1--the-game-engine)
5. [Phase 2 — Observation & State Representation](#5-phase-2--observation--state-representation)
6. [Phase 3 — The Neural Network Agents](#6-phase-3--the-neural-network-agents)
7. [Phase 4 — Training Loop & Co-Evolution](#7-phase-4--training-loop--co-evolution)
8. [Phase 5 — Visualization, Evaluation & Deployment](#8-phase-5--visualization-evaluation--deployment)
9. [Repository Structure](#9-repository-structure)
10. [Technology Stack](#10-technology-stack)
11. [Key Design Decisions](#11-key-design-decisions)

---

## 1. Project Vision

In the classic Snake game, the "bait" (food) is a passive, randomly placed item. **NeuroEvasion** fundamentally changes this:

| Classic Snake | NeuroEvasion |
|---|---|
| Bait is **static** — random spawn | Bait is an **intelligent agent** — it moves to survive |
| Snake only learns to navigate | **Both** agents learn adversarial strategies |
| Single-agent problem | **Multi-agent, zero-sum** game |
| Terminates when snake eats bait | Creates a **co-evolutionary arms race** |

The result is a system where the **Snake** (pursuer) learns to predict and intercept, and the **Bait** (evader) learns to foresee and escape — an adversarial co-evolution that teaches students the core principles of deep reinforcement learning, game theory, and emergent intelligence.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NeuroEvasion System                      │
│                                                             │
│  ┌──────────────┐     ┌──────────────┐    ┌──────────────┐  │
│  │  Game Engine │ ◄──►│  Environment │◄──►│  Renderer    │  │
│  │  (Core Loop) │     │  (Gym-like)  │    │  (Pygame)    │  │
│  └──────┬───────┘     └──────┬───────┘    └──────────────┘  │
│         │                   │                               │
│         ▼                   ▼                               │
│  ┌──────────────────────────────────────┐                   │
│  │         State Encoder                │                   │
│  │  (Grid → Tensor Observations)        │                   │
│  └──────────────┬───────────────────────┘                   │
│                 │                                           │
│        ┌────────┴────────┐                                  │
│        ▼                 ▼                                  │
│  ┌───────────┐    ┌───────────┐                             │
│  │  Snake    │    │   Bait    │                             │
│  │  Agent    │    │   Agent   │                             │
│  │  (DQN)    │    │   (DQN)   │                             │
│  └─────┬─────┘    └─────┬─────┘                             │
│        │                │                                   │
│        ▼                ▼                                   │
│  ┌──────────────────────────────────────┐                   │
│  │         Training Orchestrator        │                   │
│  │  (Self-Play / Co-Evolution Loop)     │                   │
│  └──────────────────────────────────────┘                   │
│                                                             │
│  ┌──────────────────────────────────────┐                   │
│  │    Metrics, Logging & TensorBoard    │                   │
│  └──────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 0 — Foundation & Environment

> **Goal:** Set up the development environment, project scaffolding, and all tooling.

### Steps

| # | Task | Details |
|---|------|---------|
| 0.1 | **Python Environment** | Create a `venv` with Python ≥ 3.10. Use `pyproject.toml` for dependency management. |
| 0.2 | **Install Core Dependencies** | `pygame`, `numpy`, `torch` (PyTorch), `tensorboard`, `matplotlib`, `pytest`. |
| 0.3 | **Project Scaffolding** | Create the directory structure (see §9). |
| 0.4 | **Configuration System** | Build `config.py` with dataclasses for all hyperparameters (grid size, learning rate, epsilon decay, etc.). All magic numbers live here — nowhere else. |
| 0.5 | **Git & CI** | Initialize git flow. Add pre-commit hooks for linting (`ruff`) and formatting (`black`). |
| 0.6 | **Reproducibility Seed** | Implement a `set_global_seed(seed)` utility that seeds `random`, `numpy`, and `torch` for reproducible experiments. |

### Deliverables
- [ ] Working virtual environment with all dependencies
- [ ] Complete directory tree created
- [ ] `config.py` with all default hyperparameters
- [ ] Seeding utility functional

---

## 4. Phase 1 — The Game Engine

> **Goal:** Build a clean, headless game engine that can run thousands of episodes per second without rendering.

### 1.1 Grid World

- **Discrete N×N grid** (default 20×20).
- Each cell is one of: `EMPTY`, `SNAKE_HEAD`, `SNAKE_BODY`, `BAIT`, `WALL`.
- Walls enclose the border; the arena is `(N−2) × (N−2)` playable cells.

### 1.2 Snake Mechanics

| Property | Description |
|----------|-------------|
| Movement | 4 discrete actions: `UP`, `DOWN`, `LEFT`, `RIGHT` |
| Growth | Snake grows by 1 segment when it captures the bait |
| Self-collision | Episode terminates if head hits body |
| Wall collision | Episode terminates if head hits wall |
| Turning constraint | Cannot reverse direction (e.g., moving `UP` cannot go `DOWN`) |

### 1.3 Bait Mechanics

| Property | Description |
|----------|-------------|
| Movement | 4 discrete actions: `UP`, `DOWN`, `LEFT`, `RIGHT` (+ optionally `STAY`) |
| Speed | Moves once per `k` snake steps (configurable; `k=1` for equal speed) |
| Collision rule | If bait moves into wall → stays in place (survives) |
| Capture | Bait is "caught" when snake head occupies the same cell |

### 1.4 Game Loop (Per Step)

```
1. Snake selects action  →  snake moves
2. Check collisions      →  if snake dies → game over (bait wins)
3. Bait selects action   →  bait moves  
4. Check capture         →  if snake_head == bait_pos → game over (snake wins)
5. Update state          →  return observations, rewards, done
```

### 1.5 Reward Design (Critical)

The reward function drives all learning. This is a **zero-sum** game:

| Event | Snake Reward | Bait Reward |
|-------|-------------|-------------|
| Snake captures bait | **+10.0** | **−10.0** |
| Snake dies (wall/self) | **−10.0** | **+10.0** |
| Per-step survival | **−0.01** (urgency to catch) | **+0.01** (reward for surviving) |
| Distance decreasing (snake→bait) | **+0.1** | **−0.1** |
| Distance increasing (snake→bait) | **−0.1** | **+0.1** |
| Max steps reached (timeout) | **−1.0** | **+5.0** (bait survived) |

> **Key Insight for Students**: The per-step penalties/rewards create *urgency* — the snake cannot just wander, and the bait is rewarded for simply staying alive. The distance shaping rewards provide a curriculum signal so agents don't rely only on sparse terminal rewards.

### Deliverables
- [ ] `game/grid.py` — Grid world data structure
- [ ] `game/snake.py` — Snake entity with movement and collision
- [ ] `game/bait.py` — Bait entity with movement
- [ ] `game/engine.py` — Game loop, reward computation, episode management
- [ ] Unit tests for all game mechanics

---

## 5. Phase 2 — Observation & State Representation

> **Goal:** Encode the raw game state into tensors that neural networks can consume.

### 2.1 Observation Space Design

We provide each agent with a **multi-channel 2D grid** observation (like an image with channels):

#### Snake's Observation (4 channels × N × N)
| Channel | Description |
|---------|-------------|
| 0 | **Snake body** — 1.0 where body exists, 0.0 elsewhere |
| 1 | **Snake head** — 1.0 at head position |
| 2 | **Bait position** — 1.0 at bait position |
| 3 | **Walls** — 1.0 at wall positions |

#### Bait's Observation (4 channels × N × N)
| Channel | Description |
|---------|-------------|
| 0 | **Snake body** — 1.0 where body exists |
| 1 | **Snake head** — 1.0 at head position |
| 2 | **Self (bait) position** — 1.0 at own position |
| 3 | **Walls** — 1.0 at wall positions |

### 2.2 Why Multi-Channel Grids?

> **Teaching Point:** This representation is analogous to how a CNN processes an RGB image (3 channels). Each channel encodes a different semantic feature of the game state. This is more powerful than a flat vector because it preserves **spatial relationships** — the CNN kernels can learn local patterns like "snake head approaching from the left."

### 2.3 Optional: Stacked Frames

To give agents a sense of **motion and direction**, stack the last `k` observations (default `k=3`) along the channel dimension:

```
Final observation shape: (k × 4) channels × N × N = 12 × 20 × 20
```

This lets the network infer velocity and trajectory without needing recurrent layers.

### 2.4 Gymnasium-Compatible Environment Wrapper

Wrap the game engine in a class that follows the `gymnasium` API:

```python
class NeuroEvasionEnv:
    def reset() -> (snake_obs, bait_obs)
    def step(snake_action, bait_action) -> (snake_obs, bait_obs, snake_reward, bait_reward, done, info)
    
    @property
    def observation_space  # gymnasium.spaces.Box
    
    @property
    def action_space       # gymnasium.spaces.Discrete(4)
```

### Deliverables
- [ ] `environment/state_encoder.py` — Grid-to-tensor conversion
- [ ] `environment/env.py` — Gymnasium-compatible environment wrapper
- [ ] `environment/frame_stacker.py` — Frame stacking utility
- [ ] Unit tests for observation shapes and content

---

## 6. Phase 3 — The Neural Network Agents

> **Goal:** Implement the DQN (Deep Q-Network) architecture and the agent logic for both snake and bait.

### 3.1 Network Architecture — `DQNNetwork`

```
Input: (batch_size, channels, H, W)  e.g., (32, 12, 20, 20)
                    │
                    ▼
        ┌──────────────────────┐
        │  Conv2d(channels, 32, 3×3, stride=1, pad=1)  │
        │  BatchNorm2d(32)                              │
        │  ReLU                                         │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │  Conv2d(32, 64, 3×3, stride=1, pad=1)        │
        │  BatchNorm2d(64)                              │
        │  ReLU                                         │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │  Conv2d(64, 64, 3×3, stride=1, pad=1)        │
        │  BatchNorm2d(64)                              │
        │  ReLU                                         │
        └──────────┬───────────┘
                   ▼
               Flatten
                   │
                   ▼
        ┌──────────────────────┐
        │  Linear(64*H*W, 512) │
        │  ReLU                │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │  Linear(512, num_actions)  → Q-values          │
        └──────────────────────┘

Output: Q(s, a) for each action  →  shape (batch_size, 4)
```

> **Teaching Point:** Each output neuron represents the estimated future cumulative reward for taking that action in the current state. The agent picks `argmax(Q)` during exploitation.

### 3.2 Dueling DQN Extension (Phase 3b — Optional Enhancement)

Split the final layers into **Value** and **Advantage** streams:

```
                Flatten
                   │
            ┌──────┴──────┐
            ▼              ▼
     ┌─────────────┐ ┌──────────────┐
     │ Value Stream│ │Advantage     │
     │ FC(512)     │ │Stream FC(512)│
     │ → FC(1)     │ │→ FC(actions) │
     │    V(s)     │ │   A(s,a)     │
     └──────┬──────┘ └──────┬───────┘
            │               │
            └───────┬───────┘
                    ▼
        Q(s,a) = V(s) + A(s,a) − mean(A)
```

### 3.3 Agent Class — `DQNAgent`

```python
class DQNAgent:
    def __init__(self, config):
        self.policy_net    # The network we train
        self.target_net    # Frozen copy, synced periodically
        self.memory        # Replay buffer
        self.epsilon       # Exploration rate
    
    def select_action(obs) → action        # ε-greedy policy
    def store_transition(s, a, r, s', done) # Save to replay buffer
    def train_step() → loss                 # Sample batch, compute loss, backprop
    def sync_target_network()               # Copy weights to target net
    def save(path) / load(path)             # Checkpoint management
```

### 3.4 Experience Replay Buffer

```python
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
    
    def push(state, action, reward, next_state, done)
    def sample(batch_size) → batch
    def __len__()
```

> **Teaching Point:** Replay buffers break temporal correlations in training data. Without them, consecutive game frames are highly correlated and the network overfits to recent experience. Sampling random mini-batches provides i.i.d.-like training data.

### 3.5 Epsilon-Greedy Exploration Schedule

```
ε starts at 1.0 (100% random actions)
ε decays linearly to 0.01 over 100,000 steps
ε remains at 0.01 thereafter (1% random exploration)
```

### 3.6 Loss Function — Temporal Difference (TD) Learning

```
target = r + γ × max_a' Q_target(s', a')    if not done
target = r                                    if done

loss = MSE(Q_policy(s, a),  target)
```

Where `γ = 0.99` is the discount factor.

### Deliverables
- [ ] `agents/networks.py` — DQN and Dueling DQN architectures
- [ ] `agents/dqn_agent.py` — Agent class with ε-greedy, training, checkpointing
- [ ] `agents/replay_buffer.py` — Experience replay implementation
- [ ] Unit tests for network output shapes, buffer sampling

---

## 7. Phase 4 — Training Loop & Co-Evolution

> **Goal:** Orchestrate the adversarial training where both agents improve against each other.

### 4.1 The Training Orchestrator

```python
def train(config):
    env = NeuroEvasionEnv(config)
    snake_agent = DQNAgent(config, role="snake")
    bait_agent  = DQNAgent(config, role="bait")
    
    for episode in range(config.num_episodes):        # e.g., 500,000
        snake_obs, bait_obs = env.reset()
        
        for step in range(config.max_steps):          # e.g., 200
            snake_action = snake_agent.select_action(snake_obs)
            bait_action  = bait_agent.select_action(bait_obs)
            
            (snake_obs_next, bait_obs_next,
             snake_reward, bait_reward,
             done, info) = env.step(snake_action, bait_action)
            
            snake_agent.store_transition(snake_obs, snake_action, snake_reward, snake_obs_next, done)
            bait_agent.store_transition(bait_obs, bait_action, bait_reward, bait_obs_next, done)
            
            snake_loss = snake_agent.train_step()
            bait_loss  = bait_agent.train_step()
            
            snake_obs, bait_obs = snake_obs_next, bait_obs_next
            
            if done:
                break
        
        # Periodic target network sync
        if episode % config.target_sync_interval == 0:
            snake_agent.sync_target_network()
            bait_agent.sync_target_network()
        
        # Logging
        log_metrics(episode, snake_loss, bait_loss, info)
```

### 4.2 Co-Evolutionary Dynamics

> **Teaching Point:** This is the most fascinating aspect of the system. As the snake gets better at catching the bait, the bait's training signal improves because it faces a stronger adversary. Conversely, as the bait gets better at escaping, the snake must develop more sophisticated pursuit strategies. This creates an **arms race** — a co-evolutionary loop.

**Potential Pitfalls & Solutions:**

| Problem | Description | Solution |
|---------|-------------|----------|
| **Mode Collapse** | One agent dominates, the other never learns | **Opponent Sampling** — occasionally play against past versions of the opponent (saved checkpoints) |
| **Oscillation** | Strategies cycle without improvement | **Elo Rating** system to track true skill progression |
| **Catastrophic Forgetting** | Agent forgets how to counter old strategies | Larger replay buffer + periodic play against historical opponents |

### 4.3 Opponent Sampling (Self-Play Variant)

```python
# Every K episodes, swap the live opponent with a historical checkpoint
if episode % config.opponent_swap_interval == 0:
    opponent_checkpoint = random.choice(checkpoint_history)
    bait_agent.load(opponent_checkpoint)   # or snake_agent.load(...)
```

### 4.4 Hyperparameters Summary

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `grid_size` | 20 | N×N grid dimensions |
| `num_episodes` | 500,000 | Total training episodes |
| `max_steps_per_episode` | 200 | Steps before timeout |
| `batch_size` | 64 | Replay buffer sample size |
| `learning_rate` | 1e-4 | Adam optimizer LR |
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Final exploration rate |
| `epsilon_decay_steps` | 100,000 | Steps for ε to reach minimum |
| `replay_buffer_size` | 100,000 | Maximum replay transitions |
| `target_sync_interval` | 1,000 | Episodes between target net sync |
| `frame_stack` | 3 | Number of stacked observation frames |
| `checkpoint_interval` | 10,000 | Episodes between model saves |

### Deliverables
- [ ] `training/trainer.py` — Main training orchestrator
- [ ] `training/logger.py` — TensorBoard and CSV logging
- [ ] `training/opponent_pool.py` — Historical opponent sampling
- [ ] Functional end-to-end training for 1,000 episodes as a smoke test

---

## 8. Phase 5 — Visualization, Evaluation & Deployment

> **Goal:** Make the learned behaviors visible, measure agent skill, and package everything.

### 5.1 Pygame Renderer

- Real-time visualization of trained agents playing against each other.
- Color-coded grid: snake in green gradient, bait as a pulsing red dot, trails fading.
- HUD overlay: episode count, score, epsilon, Q-value heatmap.
- Keyboard controls: `P` pause, `S` step-by-step, `R` restart, `+/-` speed control.

### 5.2 Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Win Rate** | % of episodes snake captures bait (over rolling 1000 games) |
| **Average Survival Time** | Mean # of steps bait survives |
| **Elo Rating** | Skill progression for both agents over training |
| **Average Q-value** | Confidence of agents (detect over/under-estimation) |
| **Average Episode Reward** | Learning stability indicator |
| **Entropy of Actions** | How random/deterministic the policy has become |

### 5.3 Visualization Suite

- **Training Curves** — Loss, reward, win-rate over episodes (matplotlib / TensorBoard)
- **Strategy Heatmaps** — Where does the bait tend to go? Where does the snake tend to aim?
- **Q-Value Landscape** — Visual grid showing Q-values for each cell/action
- **Emergent Behavior GIFs** — Recorded episodes showing learned strategies

### 5.4 Demo Mode

```bash
python main.py demo --snake-model checkpoints/snake_best.pt --bait-model checkpoints/bait_best.pt
```

Runs the game with pre-trained agents in a Pygame window with full HUD.

### Deliverables
- [ ] `visualization/renderer.py` — Pygame game renderer
- [ ] `visualization/hud.py` — Stats overlay
- [ ] `visualization/plots.py` — Training curve plotting
- [ ] `evaluation/evaluator.py` — Win rate, Elo, metrics computation
- [ ] `main.py` — CLI entry point (`train`, `demo`, `evaluate` modes)

---

## 9. Repository Structure

```
NeuroEvasion/
│
├── main.py                     # CLI entry point
├── config.py                   # All hyperparameters (dataclass)
├── requirements.txt            # Dependencies
├── pyproject.toml              # Project metadata
├── README.md                   # Project overview
├── LICENSE                     # License file
│
├── game/                       # Phase 1: Core Game Engine
│   ├── __init__.py
│   ├── grid.py                 # Grid world data structure
│   ├── snake.py                # Snake entity
│   ├── bait.py                 # Bait entity
│   └── engine.py               # Game loop & reward computation
│
├── environment/                # Phase 2: RL Environment
│   ├── __init__.py
│   ├── env.py                  # Gymnasium-like wrapper
│   ├── state_encoder.py        # Grid → tensor conversion
│   └── frame_stacker.py        # Frame stacking utility
│
├── agents/                     # Phase 3: Neural Network Agents
│   ├── __init__.py
│   ├── networks.py             # DQN / Dueling DQN architectures
│   ├── dqn_agent.py            # Agent logic (ε-greedy, training)
│   └── replay_buffer.py        # Experience replay
│
├── training/                   # Phase 4: Training Orchestration
│   ├── __init__.py
│   ├── trainer.py              # Main training loop
│   ├── logger.py               # TensorBoard + CSV logging
│   └── opponent_pool.py        # Historical opponent sampling
│
├── visualization/              # Phase 5: Rendering & Plots
│   ├── __init__.py
│   ├── renderer.py             # Pygame renderer
│   ├── hud.py                  # HUD overlay
│   └── plots.py                # Matplotlib training curves
│
├── evaluation/                 # Phase 5: Evaluation
│   ├── __init__.py
│   └── evaluator.py            # Metrics computation
│
├── checkpoints/                # Saved model weights
│   └── .gitkeep
│
├── logs/                       # TensorBoard logs
│   └── .gitkeep
│
├── tests/                      # Unit & integration tests
│   ├── test_grid.py
│   ├── test_snake.py
│   ├── test_bait.py
│   ├── test_engine.py
│   ├── test_env.py
│   ├── test_networks.py
│   ├── test_agent.py
│   ├── test_replay_buffer.py
│   └── test_training.py
│
└── docs/                       # Documentation
    ├── ARCHITECTURE.md          # ← This document
    └── TUTORIAL.md              # Full implementation tutorial
```

---

## 10. Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Language** | Python 3.10+ | Industry standard for ML/DL |
| **Deep Learning** | PyTorch | Dynamic computation graphs, best for research/education |
| **Game Rendering** | Pygame | Lightweight, perfect for grid-based visualization |
| **Numerics** | NumPy | Fast array operations for grids and observations |
| **RL Environment** | Custom (Gymnasium-style API) | Teaches students the standard RL interface |
| **Logging** | TensorBoard | Real-time training metric visualization |
| **Plotting** | Matplotlib | Publication-quality training curves |
| **Testing** | pytest | Clean, powerful test framework |
| **Linting** | Ruff + Black | Code quality and consistent formatting |

---

## 11. Key Design Decisions

### Why DQN over Policy Gradient methods?

DQN is chosen as the **primary algorithm** because:
1. **Pedagogical clarity** — Students can understand Q-values as a table of "how good is action A in state S"
2. **Discrete action spaces** — Our 4-directional movement maps perfectly
3. **Stable training** — With target networks and replay buffers, DQN trains reliably
4. **Debuggability** — You can inspect Q-values directly to understand agent reasoning

### Why not a single network for both agents?

Separate networks for snake and bait ensure:
- Each agent has its own replay buffer (different reward signals)
- Independent learning rates and exploration schedules
- Clean conceptual separation for students

### Why grid-based observations instead of feature vectors?

Multi-channel grids:
- Preserve **spatial structure** (essential for pursuit-evasion)
- Enable **convolutional layers** (students learn CNNs in context)
- Scale naturally if grid size changes
- Analogous to image-based RL (Atari) — transfer of concepts

### Why zero-sum reward design?

Zero-sum (snake reward = −bait reward) directly encodes the adversarial nature:
- What's good for one agent is bad for the other
- Prevents both agents from finding collaborative "easy" solutions
- Creates genuine competitive pressure that drives co-evolution

---

> **Next Step:** Proceed to [TUTORIAL.md](./TUTORIAL.md) for the complete, step-by-step implementation guide with code.
]]>
