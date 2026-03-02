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
