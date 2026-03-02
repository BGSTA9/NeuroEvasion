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
