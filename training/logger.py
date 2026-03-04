"""
training/logger.py — Training metrics logging.

Logs to both TensorBoard (interactive graphs) and CSV (raw data).
TensorBoard lets you monitor training in real-time with:
    tensorboard --logdir logs/

RESUME SUPPORT:
    When training is resumed from a checkpoint, call resume_from() instead of
    creating a new TrainingLogger. This reattaches to the SAME log directory so:
        - TensorBoard graphs are continuous (no second run overlapping the first)
        - The CSV file gains new rows rather than being overwritten
"""

import os
import csv
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """
    Dual logger: TensorBoard + CSV file.

    Usage (fresh run):
        logger = TrainingLogger("logs/run_001")
        logger.log_episode(episode=100, snake_reward=5.2, ...)
        logger.close()

    Usage (resumed run):
        logger = TrainingLogger.resume_from("logs/run_001", start_episode=5000)
        logger.log_episode(episode=5001, ...)
        logger.close()
    """

    def __init__(self, log_dir: str, _append_csv: bool = False) -> None:
        """
        Initialise a fresh logger.

        Args:
            log_dir:        Directory for TensorBoard events and metrics.csv.
            _append_csv:    Internal flag — set True by resume_from() so the
                            CSV is opened in append mode instead of overwrite.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        csv_path = os.path.join(log_dir, "metrics.csv")
        csv_exists = os.path.exists(csv_path)

        if _append_csv and csv_exists:
            # Append mode — do NOT write the header row again
            self.csv_file = open(csv_path, "a", newline="")
            self.csv_writer = csv.writer(self.csv_file)
        else:
            # Fresh write — overwrite any stale file and write header
            self.csv_file = open(csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "episode", "snake_reward", "bait_reward", "snake_loss",
                "bait_loss", "winner", "steps", "epsilon", "timestamp"
            ])

    @classmethod
    def resume_from(cls, log_dir: str, start_episode: int) -> "TrainingLogger":
        """
        Reattach to an existing log directory for a resumed training run.

        The TensorBoard SummaryWriter will pick up from where it left off
        (events are appended to the same directory), and the CSV is opened
        in append mode so all episodes end up in a single, continuous file.

        Args:
            log_dir:        The SAME log directory used by the interrupted run.
                            Found in training_state.json → "log_dir".
            start_episode:  Episode number we are resuming from (for logging).

        Returns:
            A TrainingLogger ready to accept log_episode() calls.

        Raises:
            FileNotFoundError: If log_dir does not exist on disk.
        """
        if not os.path.isdir(log_dir):
            raise FileNotFoundError(
                f"Log directory not found: '{log_dir}'. "
                "Cannot resume — the log directory may have been deleted. "
                "Create a fresh TrainingLogger to start a new run."
            )
        instance = cls(log_dir, _append_csv=True)
        # Write a resume marker so TensorBoard timelines stay readable
        instance.writer.add_text(
            "Training/Events",
            f"▶️ Resumed from episode {start_episode} at {datetime.now().isoformat()}",
            global_step=start_episode,
        )
        return instance

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

        snake_win = 1.0 if winner == "capture" else 0.0
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
