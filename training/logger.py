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
