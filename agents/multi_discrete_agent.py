"""
agents/multi_discrete_agent.py — DQN agent for the Multi-Discrete action space.

MULTI-DISCRETE DQN:
    Extends DQNAgent with a dual Q-head:

        - move_head  → argmax over movement actions
        - tool_head  → argmax over tool-use actions

    Both heads are trained simultaneously via separate TD losses but share
    the same CNN backbone (gradient flows back through both heads).

COEVOLUTIONARY FIXES (v2):
    - Double DQN: policy net selects action, target net evaluates (both heads)
    - Polyak soft target update (τ=0.005): smooth tracking, no more hard sync
    - Huber loss replaces MSE: robust to large TD-errors
    - gradient clipping at max_norm=1.0 (tightened from 10.0)
    - Cosine-annealing LR scheduler with warm restarts
    - Cyclic epsilon (inherited from DQNAgent._update_epsilon)

CHECKPOINT COMPATIBILITY:
    get_full_state() / load_full_state() add "move_head" / "tool_head"
    keys in addition to the base "policy_net" / "target_net" keys. The
    CheckpointManager stores/loads these transparently.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn_agent import DQNAgent
from agents.networks import MultiDiscreteNetwork, MultiDiscreteDuelingNetwork
from agents.replay_buffer import ReplayBuffer
from game.actions import MultiDiscreteAction


class MultiDiscreteDQNAgent(DQNAgent):
    """
    Deep Q-Network agent for a Multi-Discrete (movement × tool-use) action space.

    Inherits from DQNAgent for:
        - epsilon schedule and _update_epsilon() (including cyclic ε)
        - replay buffer management
        - soft_update_target() (Polyak averaging)
        - checkpoint save / load wrappers

    Overrides:
        - __init__    — builds dual-head networks instead of single-head
        - select_action → MultiDiscreteAction
        - train_step  → dual-loss backprop with Double DQN
        - get_full_state / load_full_state — multi-head state dicts

    Args:
        in_channels:       observation channels (frame_stack × 4)
        grid_size:         grid height/width
        num_move_actions:  number of movement options (4 snake, 5 bait)
        num_tool_actions:  number of tool options (3 by default)
        config:            AgentConfig dataclass
        device:            'cpu' or 'cuda'
        use_dueling:       use Dueling decomposition in both heads
    """

    def __init__(
        self,
        in_channels: int,
        grid_size: int,
        num_move_actions: int,
        num_tool_actions: int,
        config,
        device: str = "cpu",
        use_dueling: bool = False,
    ):
        # ------------------------------------------------------------------
        # We deliberately bypass DQNAgent.__init__ because we need different
        # network classes. We replicate only what we need.
        # ------------------------------------------------------------------
        # Replicate base agent state (no super().__init__ call)
        self.config = config
        self.device = torch.device(device)

        # ── Action space info ──────────────────────────────────────────────
        self.num_move_actions = num_move_actions
        self.num_tool_actions = num_tool_actions
        # DQNAgent expects this for `select_action` fallback path
        self.num_actions = num_move_actions * num_tool_actions

        # ── Exploration ─────────────────────────────────────────────────────
        self.epsilon = config.epsilon_start
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay_steps = config.epsilon_decay_steps
        self.steps_done = 0

        # ── Replay buffer ───────────────────────────────────────────────────
        # Actions are encoded as single ints via MultiDiscreteAction.encode()
        self.memory = ReplayBuffer(capacity=config.replay_buffer_size)

        # ── Networks ────────────────────────────────────────────────────────
        NetworkClass = MultiDiscreteDuelingNetwork if use_dueling else MultiDiscreteNetwork

        self.policy_net = NetworkClass(
            in_channels, grid_size, num_move_actions, num_tool_actions
        ).to(self.device)

        self.target_net = NetworkClass(
            in_channels, grid_size, num_move_actions, num_tool_actions
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # ── Optimizer ───────────────────────────────────────────────────────
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.learning_rate
        )

        # ── LR Scheduler (cosine annealing with warm restarts) ──────────────
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=100_000,
            T_mult=2,
            eta_min=getattr(config, 'lr_min', 1e-6),
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Action selection
    # ──────────────────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> MultiDiscreteAction:
        """
        ε-greedy Multi-Discrete action selection.

        With probability ε: sample both sub-actions uniformly at random.
        With probability 1-ε: take argmax over each Q-head independently.

        WHY INDEPENDENT ARGMAX?
            If we assume movement and tool are conditionally independent given
            the state, argmax-ing each head separately is the correct greedy
            policy for a factored action space. In practice the shared CNN
            backbone introduces soft coupling between the two decisions.

        Returns:
            MultiDiscreteAction(move=..., tool=...)
        """
        self.steps_done += 1
        self._update_epsilon()

        if np.random.random() < self.epsilon:
            # Explore: sample each sub-action uniformly
            return MultiDiscreteAction(
                move=np.random.randint(self.num_move_actions),
                tool=np.random.randint(self.num_tool_actions),
            )

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            move_q, tool_q = self.policy_net(state_t)
            move_act = move_q.argmax(dim=1).item()
            tool_act = tool_q.argmax(dim=1).item()
            return MultiDiscreteAction(move=move_act, tool=tool_act)

    # ──────────────────────────────────────────────────────────────────────────
    #  Storing transitions
    # ──────────────────────────────────────────────────────────────────────────

    def store_transition(
        self,
        state: np.ndarray,
        action: MultiDiscreteAction | int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Encode the (move, tool) pair into a single int before storing.

        Accepts either a MultiDiscreteAction or a plain int (if the latter,
        it's stored as-is to maintain backward compatibility with any legacy
        call sites).
        """
        if isinstance(action, MultiDiscreteAction):
            encoded = action.encode(self.num_tool_actions)
        else:
            encoded = int(action)
        self.memory.push(state, encoded, reward, next_state, done)

    # ──────────────────────────────────────────────────────────────────────────
    #  Training step
    # ──────────────────────────────────────────────────────────────────────────

    def train_step(self) -> float | None:
        """
        Perform one gradient update on the dual-head network.

        MULTI-DISCRETE DOUBLE DQN LOSSES:
            For each head (move / tool):
                1. Policy net selects the best next action:
                       a* = argmax_a Q_policy_head(s', a)
                2. Target net evaluates that action:
                       Q_val = Q_target_head(s', a*)
                3. TD target: y = r + γ × Q_val × (1 - done)
                4. Loss: Huber(Q_head(s, a_taken), y)

            Total loss = L_move + L_tool

            This trains the shared CNN backbone on both objectives jointly,
            while each head's weights receive gradients from its own loss.

        Returns:
            Combined loss value (float), or None if buffer isn't ready.
        """
        if not self.memory.is_ready(self.config.batch_size):
            return None

        batch = self.memory.sample(self.config.batch_size)

        states      = torch.FloatTensor(batch["states"]).to(self.device)
        encoded_acts = torch.LongTensor(batch["actions"]).to(self.device)
        rewards     = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones       = torch.FloatTensor(batch["dones"]).to(self.device)

        # Decode encoded actions → (move, tool) index tensors
        move_acts = encoded_acts // self.num_tool_actions        # (batch,)
        tool_acts = encoded_acts  % self.num_tool_actions        # (batch,)

        # ── Current Q-values ───────────────────────────────────────────────
        move_q_all, tool_q_all = self.policy_net(states)
        # Gather Q(s, a) for the actual actions taken
        move_q = move_q_all.gather(1, move_acts.unsqueeze(1)).squeeze(1)
        tool_q = tool_q_all.gather(1, tool_acts.unsqueeze(1)).squeeze(1)

        # ── Target Q-values (Double DQN) ──────────────────────────────────
        with torch.no_grad():
            use_double = getattr(self.config, 'use_double_dqn', True)

            if use_double:
                # Policy net SELECTS the best next actions
                next_move_q_policy, next_tool_q_policy = self.policy_net(next_states)
                best_move_acts = next_move_q_policy.argmax(dim=1)
                best_tool_acts = next_tool_q_policy.argmax(dim=1)

                # Target net EVALUATES those actions
                next_move_q_target, next_tool_q_target = self.target_net(next_states)
                next_move_val = next_move_q_target.gather(
                    1, best_move_acts.unsqueeze(1)).squeeze(1)
                next_tool_val = next_tool_q_target.gather(
                    1, best_tool_acts.unsqueeze(1)).squeeze(1)
            else:
                # Vanilla DQN fallback
                next_move_q_target, next_tool_q_target = self.target_net(next_states)
                next_move_val = next_move_q_target.max(dim=1)[0]
                next_tool_val = next_tool_q_target.max(dim=1)[0]

            # Bellman targets
            move_target = rewards + self.config.gamma * next_move_val * (1 - dones)
            tool_target = rewards + self.config.gamma * next_tool_val * (1 - dones)

        # ── Combined Huber loss + backprop ─────────────────────────────────
        loss_move = nn.functional.smooth_l1_loss(move_q, move_target)
        loss_tool = nn.functional.smooth_l1_loss(tool_q, tool_target)
        loss = loss_move + loss_tool

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=getattr(self.config, 'grad_clip', 1.0),
        )
        self.optimizer.step()

        # Step the LR scheduler
        self.scheduler.step()

        # ── Soft target update (Polyak averaging) ──────────────────────────
        if getattr(self.config, 'use_soft_target_update', True):
            self.soft_update_target()

        return loss.item()

    # ──────────────────────────────────────────────────────────────────────────
    #  Checkpoint helpers
    # ──────────────────────────────────────────────────────────────────────────

    def get_full_state(self, include_optimizer: bool = True) -> dict:
        """
        Full, resumable snapshot of this multi-discrete agent.

        Adds "num_move_actions" and "num_tool_actions" so that the agent
        can be reconstructed from scratch just from the checkpoint.
        """
        state = {
            "policy_net":       self.policy_net.state_dict(),
            "target_net":       self.target_net.state_dict(),
            "epsilon":          self.epsilon,
            "steps_done":       self.steps_done,
            "num_move_actions": self.num_move_actions,
            "num_tool_actions": self.num_tool_actions,
            "agent_type":       "MultiDiscreteDQNAgent",
        }
        if include_optimizer:
            state["optimizer"] = self.optimizer.state_dict()
            state["scheduler"] = self.scheduler.state_dict()
        return state

    def load_full_state(self, state: dict) -> None:
        """
        Restore agent from a checkpoint dict (as saved by get_full_state).

        Gracefully handles checkpoints saved without optimizer state.
        """
        self.policy_net.load_state_dict(state["policy_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.epsilon    = state["epsilon"]
        self.steps_done = state["steps_done"]
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])

    def sync_target_network(self) -> None:
        """
        Hard-copy policy network weights to target network.

        DEPRECATED for coevolutionary training — inherited soft_update_target()
        is used instead. Kept for backward compatibility.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_q_values(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (move_q_values, tool_q_values) for a state.

        Useful for visualization, analysis, and debugging.

        Returns:
            move_q: np.ndarray of shape (num_move_actions,)
            tool_q: np.ndarray of shape (num_tool_actions,)
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            move_q, tool_q = self.policy_net(state_t)
            return move_q.cpu().numpy()[0], tool_q.cpu().numpy()[0]

    # ── Thin wrappers for legacy save/load interface ───────────────────────────

    def save(self, path: str) -> None:
        """Save checkpoint to a flat .pt file (legacy interface)."""
        import os
        import torch as _torch
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        _torch.save(self.get_full_state(include_optimizer=True), path)

    def load(self, path: str) -> None:
        """Load checkpoint from a flat .pt file (legacy interface)."""
        import torch as _torch
        state = _torch.load(path, map_location=self.device, weights_only=False)
        self.load_full_state(state)
