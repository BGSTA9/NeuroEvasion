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
    7. Compute loss: L = Huber(Q(s, a) - y)   ← Huber, not MSE
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
            loss   = Huber(Q_policy(state, action), target)

        WHY HUBER LOSS (smooth_l1_loss)?
            MSE squares large errors, which causes exploding gradients when
            Q-value estimates are far off (common in co-evolutionary training
            where the opponent keeps shifting the reward landscape).
            Huber is quadratic for small errors (precise) and linear for large
            ones (robust) — it stops runaway loss spikes dead.

        WHY max_norm=grad_clip (1.0 vs old 10.0)?
            max_norm=10.0 is so loose it never actually clips anything.
            max_norm=1.0 hard-caps the gradient vector length, preventing
            a single bad batch from taking a catastrophically large step.

        Returns:
            loss value (float) or None if buffer isn't ready
        """
        if not self.memory.is_ready(self.config.batch_size):
            return None

        batch = self.memory.sample(self.config.batch_size)

        # Convert to tensors
        states      = torch.FloatTensor(batch['states']).to(self.device)
        actions     = torch.LongTensor(batch['actions']).to(self.device)
        rewards     = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones       = torch.FloatTensor(batch['dones']).to(self.device)

        # --- Current Q-values ---
        # Q(s, a) for the actions we actually took
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- Target Q-values ---
        with torch.no_grad():
            # max_a' Q_target(s', a')
            next_q = self.target_net(next_states).max(dim=1)[0]
            # If episode is done, there is no next state value
            target_q = rewards + self.config.gamma * next_q * (1 - dones)

        # --- Compute Huber loss ---
        # FIX 1: smooth_l1_loss = true Huber loss.
        # Previous mse_loss was squaring large TD errors, amplifying
        # the Q-value divergence visible from episode 25,000 onwards.
        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        # --- Backprop ---
        self.optimizer.zero_grad()
        loss.backward()

        # FIX 2: tightened gradient clip from max_norm=10.0 → config.grad_clip (1.0).
        # The old value of 10.0 was so permissive it never triggered.
        # 1.0 hard-caps the gradient vector norm on every step, stopping
        # the loss spiral that appeared once epsilon hit its floor at ep 21,700.
        nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                 max_norm=self.config.grad_clip)

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

    def get_full_state(self, include_optimizer: bool = True) -> dict:
        """
        Return a complete, resumable snapshot of this agent.

        The returned dict contains everything needed to reconstruct the
        agent at exactly the same point in training:

            policy_net      — the network being trained
            target_net      — the stable reference network
            optimizer       — Adam state (step counts, momentum buffers)
            epsilon         — current exploration rate
            steps_done      — total environment steps taken (drives ε-decay)

        Args:
            include_optimizer:  Set to False to save disk space at the cost
                                of resetting the optimizer state on resume.
        """
        state = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon":    self.epsilon,
            "steps_done": self.steps_done,
        }
        if include_optimizer:
            state["optimizer"] = self.optimizer.state_dict()
        return state

    def load_full_state(self, state: dict) -> None:
        """
        Restore agent to a previously saved state.

        Handles the case where the checkpoint was saved without optimizer
        state (e.g. when include_optimizer=False was used at save time).
        In that scenario the optimizer starts fresh but weights and epsilon
        are resumed correctly.

        Args:
            state: Dict as returned by get_full_state().
        """
        self.policy_net.load_state_dict(state["policy_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.epsilon    = state["epsilon"]
        self.steps_done = state["steps_done"]
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])

    # ─── Backward-compatible thin wrappers ───────────────────────────────────

    def save(self, path: str) -> None:
        """Save a checkpoint to a flat .pt file (legacy interface)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.get_full_state(include_optimizer=True), path)

    def load(self, path: str) -> None:
        """Load a checkpoint from a flat .pt file (legacy interface)."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_full_state(state)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state (useful for visualization/debugging)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_tensor).cpu().numpy()[0]