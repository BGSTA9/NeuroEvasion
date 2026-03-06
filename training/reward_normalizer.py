"""
training/reward_normalizer.py — Running reward normalisation.

WHY NORMALISE REWARDS?
    In coevolutionary training, the reward distribution shifts constantly:
    early on, most episodes end quickly with large terminal rewards; later,
    episodes become longer with shaping rewards dominating.

    If we feed raw rewards into the replay buffer, gradient magnitudes swing
    wildly — this is a major contributor to the loss explosion observed from
    episode ~25,000 onwards in the original run.

    Running normalisation uses Welford's online algorithm to maintain a
    running mean and standard deviation.  Each reward is centred and scaled
    before being stored, keeping gradient magnitudes in a stable range
    regardless of training phase.

    IMPORTANT: we only normalise the reward *before storing* in the replay
    buffer.  Terminal signals (done flags) are not affected.
"""

import math


class RewardNormalizer:
    """
    Online reward normaliser using Welford's algorithm.

    Attributes:
        clip:  Hard clip normalised rewards to [-clip, +clip].
        mean:  Running mean of observed rewards.
        var:   Running variance (biased).
        count: Number of observed rewards.
    """

    def __init__(self, clip: float = 10.0):
        self.clip = clip
        self.mean: float = 0.0
        self.var: float = 1.0
        self._m2: float = 0.0
        self.count: int = 0

    def normalize(self, reward: float) -> float:
        """
        Update running statistics and return the normalised reward.

        The normalised reward is clipped to [-clip, +clip] for safety.
        """
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self._m2 += delta * delta2
        self.var = self._m2 / max(self.count, 1)

        std = max(math.sqrt(self.var), 1e-8)
        normalised = (reward - self.mean) / std
        return max(-self.clip, min(self.clip, normalised))

    def get_state(self) -> dict:
        """Serialize for checkpoint resume."""
        return {
            "mean": self.mean,
            "var": self.var,
            "_m2": self._m2,
            "count": self.count,
            "clip": self.clip,
        }

    def load_state(self, state: dict) -> None:
        """Restore from checkpoint."""
        self.mean = state["mean"]
        self.var = state["var"]
        self._m2 = state["_m2"]
        self.count = state["count"]
        self.clip = state.get("clip", self.clip)
