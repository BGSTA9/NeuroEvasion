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
