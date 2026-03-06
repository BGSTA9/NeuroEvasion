"""
training/opponent_pool.py — Historical opponent snapshot pool.

WHY AN OPPONENT POOL?
    The single biggest failure mode in coevolutionary RL is the Red Queen
    effect: both agents keep adapting to each other's *current* policy,
    but neither develops robust, generalisable strategies.

    The opponent pool maintains a ring buffer of frozen opponent weights
    from earlier training.  During each episode, there is a configurable
    probability (default 30%) of playing against a randomly selected
    historical snapshot instead of the live opponent.

    This has three stabilising effects:
      1. Prevents "forgetting" — the agent must still beat older strategies.
      2. Smooths the non-stationarity — the effective opponent changes
         less abruptly between consecutive episodes.
      3. Encourages generalisation — beating many opponent variants
         produces a more robust policy than beating only the latest one.

    IMPLEMENTATION:
        Snapshots are full policy_net state_dicts (not full agent states).
        Loading a snapshot into a temporary "shadow" network is cheap —
        we do NOT touch the live agent's optimizer or epsilon state.

CHECKPOINT SUPPORT:
    The pool serialises to/from a list of state_dicts, stored alongside
    the regular checkpoint.  On resume, historical opponents are restored.
"""

import copy
import random
from typing import Any


class OpponentPool:
    """
    Ring buffer of frozen opponent policy snapshots.

    Args:
        pool_size:     Maximum number of snapshots to keep.
        current_prob:  Probability of playing the live opponent (vs. history).
    """

    def __init__(self, pool_size: int = 10, current_prob: float = 0.70):
        self.pool_size = pool_size
        self.current_prob = current_prob
        self.snapshots: list[dict[str, Any]] = []

    # ─── Snapshot management ──────────────────────────────────────────────────

    def save_snapshot(self, agent) -> None:
        """
        Deep-copy the agent's policy_net state_dict into the pool.

        If the pool is full, the oldest snapshot is evicted (FIFO).
        """
        snapshot = copy.deepcopy(agent.policy_net.state_dict())
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.pool_size:
            self.snapshots.pop(0)

    def should_use_historical(self) -> bool:
        """Return True if this episode should use a historical opponent."""
        if not self.snapshots:
            return False
        return random.random() > self.current_prob

    def get_random_snapshot(self) -> dict[str, Any]:
        """Return a random snapshot from the pool."""
        return random.choice(self.snapshots)

    # ─── Checkpoint serialisation ─────────────────────────────────────────────

    def get_state(self) -> dict:
        """Serialize the entire pool for checkpointing."""
        return {
            "snapshots": [s for s in self.snapshots],
            "pool_size": self.pool_size,
            "current_prob": self.current_prob,
        }

    def load_state(self, state: dict) -> None:
        """Restore the pool from a checkpoint."""
        self.snapshots = state.get("snapshots", [])
        self.pool_size = state.get("pool_size", self.pool_size)
        self.current_prob = state.get("current_prob", self.current_prob)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __repr__(self) -> str:
        return (f"OpponentPool(size={len(self.snapshots)}/{self.pool_size}, "
                f"current_prob={self.current_prob:.2f})")
