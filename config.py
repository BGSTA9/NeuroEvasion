"""
config.py — Central configuration for the NeuroEvasion project.

WHY A DATACLASS?
    Dataclasses give us type-checked, documented, immutable configuration.
    Every "magic number" in the project traces back to here.
"""
# ──────────────────────────────────────────────────────────────────────────────
# NOTE ON COLAB H100 USAGE
# Set training.device = "cuda" and checkpoint.drive_sync_dir to your
# mounted Google Drive path so checkpoints survive runtime restarts.
# Example:
#   config.checkpoint.drive_sync_dir = "/content/drive/MyDrive/NeuroEvasion/ckpts"
# ──────────────────────────────────────────────────────────────────────────────

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
class CheckpointConfig:
    """
    Configuration for the checkpoint & resume system.

    Attributes:
        checkpoint_dir:     Root directory for all checkpoint subdirectories.
        interval:           Save a checkpoint every N episodes.
        keep_last_n:        Keep the N most recent regular checkpoints;
                            older ones are deleted to save disk space.
        save_optimizer:     Include the Adam optimizer state so the learning
                            rate schedule and momentum are restored exactly.
        atomic_write:       Write to a .tmp file then os.replace(); guarantees
                            the previous checkpoint is never corrupted by a
                            crash mid-write.
        drive_sync_dir:     If non-empty, mirror each checkpoint to this path
                            after saving (e.g. a Google Drive mount on Colab).
                            Training continues even if the sync fails.
    """
    checkpoint_dir: str = "checkpoints"
    interval: int = 1_000             # Save every 1 000 episodes (tunable)
    keep_last_n: int = 5              # Keep 5 rolling checkpoints on disk
    save_optimizer: bool = True       # Essential for exact resume
    atomic_write: bool = True         # Crash-safe writes
    drive_sync_dir: str = ""         # Mount path for Google Drive sync


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""
    num_episodes: int = 500_000
    checkpoint_interval: int = 10_000  # Deprecated: use CheckpointConfig.interval
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
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    seed: int = 42
