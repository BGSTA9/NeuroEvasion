"""
agents/dqn_agent.py — The DQN agent that learns to play.

This class ties everything together:
    - Uses the DQN network to estimate Q-values
    - Selects actions via ε-greedy exploration
    - Stores experiences in the replay buffer
    - Trains the network using TD-learning
    - Maintains a target network for stable training

COEVOLUTIONARY FIXES (v2):
    - Double DQN: policy net selects action, target net evaluates.
      Prevents the systematic Q-value overestimation that caused
      the catastrophic divergence from episode ~25,000 onwards.
    - Polyak soft target update (τ=0.005): smooth tracking replaces
      hard copy every 1,000 episodes, eliminating "target shock".
    - Cyclic epsilon: cosine-cycled exploration re-injects diversity
      after the initial linear decay, preventing policy crystallisation
      against a non-stationary coevolutionary opponent.
    - Cosine-annealing LR with warm restarts: adapts learning rate
      to the training phase instead of a fixed value forever.

THE TRAINING ALGORITHM (pseudocode):
    1. Observe state s
    2. With probability ε: random action, else: argmax Q(s, a)
    3. Execute action, observe reward r and next state s'
    4. Store (s, a, r, s', done) in replay buffer
    5. Sample random batch from buffer
    6. Double DQN target:
         a* = argmax_a Q_policy(s', a)    ← policy selects
         y  = r + γ × Q_target(s', a*)    ← target evaluates
    7. Compute loss: L = Huber(Q(s, a) - y)
    8. Backpropagate and update weights
    9. Soft-update target network (Polyak τ=0.005)
"""

import math
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

        # --- LR Scheduler (cosine annealing with warm restarts) ---
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=100_000,           # First restart after 100k steps
            T_mult=2,              # Double the period after each restart
            eta_min=config.lr_min,
        )

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

        WHY CYCLIC ε?
            In coevolutionary training, the opponent keeps changing.
            A fixed-floor ε means the agent stops exploring exactly when
            it needs to adapt most. Cycling ε back up periodically
            re-injects exploration and prevents policy crystallisation.
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
        """
        Update ε with initial linear decay → cosine cycling.

        Phase 1 (steps < decay_steps):
            Linear decay from epsilon_start → epsilon_end.

        Phase 2 (steps >= decay_steps, if epsilon_cycle enabled):
            Cosine cycling between epsilon_cycle_min and epsilon_cycle_max.
            This ensures the agent never fully commits to exploitation,
            which is critical in coevolutionary settings.
        """
        progress = min(1.0, self.steps_done / self.epsilon_decay_steps)
        base = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

        if (hasattr(self.config, 'epsilon_cycle') and
                self.config.epsilon_cycle and progress >= 1.0):
            # Phase 2: Cosine cycling
            excess_steps = self.steps_done - self.epsilon_decay_steps
            period = getattr(self.config, 'epsilon_cycle_period', 200_000)
            cycle_min = getattr(self.config, 'epsilon_cycle_min', 0.05)
            cycle_max = getattr(self.config, 'epsilon_cycle_max', 0.30)

            cycle_progress = excess_steps / period
            cosine = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
            base = cycle_min + cosine * (cycle_max - cycle_min)

        self.epsilon = base

    def store_transition(self, state, action, reward, next_state, done) -> None:
        """Save an experience to replay memory."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """
        Perform one training step (sample batch + gradient update).

        DOUBLE DQN UPDATE (fixes Q-value overestimation):
            1. Policy net selects the best next action:
                   a* = argmax_a Q_policy(s', a)
            2. Target net evaluates that action:
                   Q_target_val = Q_target(s', a*)
            3. TD target:
                   y = r + γ × Q_target_val × (1 - done)

            Standard DQN uses max Q_target(s', ·) for BOTH selection and
            evaluation, which systematically overestimates Q-values. The
            overestimates compound through bootstrapping, eventually causing
            the divergence we observed from episode ~25,000 onwards.

        WHY HUBER LOSS (smooth_l1_loss)?
            MSE squares large errors, which causes exploding gradients when
            Q-value estimates are far off (common in co-evolutionary training
            where the opponent keeps shifting the reward landscape).
            Huber is quadratic for small errors (precise) and linear for large
            ones (robust) — it stops runaway loss spikes dead.

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

        # --- Target Q-values (Double DQN) ---
        with torch.no_grad():
            use_double = getattr(self.config, 'use_double_dqn', True)
            if use_double:
                # Double DQN: policy net SELECTS, target net EVALUATES
                best_actions = self.policy_net(next_states).argmax(dim=1)
                next_q = self.target_net(next_states).gather(
                    1, best_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                # Vanilla DQN fallback
                next_q = self.target_net(next_states).max(dim=1)[0]

            # If episode is done, there is no next state value
            target_q = rewards + self.config.gamma * next_q * (1 - dones)

        # --- Compute Huber loss ---
        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        # --- Backprop ---
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (max_norm=1.0)
        nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                 max_norm=self.config.grad_clip)

        self.optimizer.step()

        # Step the LR scheduler
        self.scheduler.step()

        # --- Soft target update (Polyak averaging) ---
        if getattr(self.config, 'use_soft_target_update', True):
            self.soft_update_target()

        return loss.item()

    def soft_update_target(self, tau: float | None = None) -> None:
        """
        Polyak-average target network towards policy network.

        θ_target ← τ × θ_policy + (1 − τ) × θ_target

        WHY SOFT UPDATES?
            Hard copy (every N episodes) creates "target shock" — the
            target Q-values jump discontinuously, destabilising all
            TD-error computations for subsequent batches.

            Soft update moves the target a tiny bit (τ=0.005 = 0.5%)
            towards the policy on EVERY training step. This gives the
            target network the stability of a slow-moving reference
            while still tracking the policy's improvements.

        This is the method used by DDPG, TD3, SAC, and other modern
        deep RL algorithms.

        Args:
            tau: Override for the soft-update coefficient.
                 Defaults to config.target_update_tau.
        """
        if tau is None:
            tau = getattr(self.config, 'target_update_tau', 0.005)
        for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )

    def sync_target_network(self) -> None:
        """
        Hard-copy policy network weights to target network.

        DEPRECATED for coevolutionary training — use soft_update_target().
        Kept for backward compatibility and non-coevolutionary settings.
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
            scheduler       — LR scheduler state
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
            state["scheduler"] = self.scheduler.state_dict()
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
        if "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])

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