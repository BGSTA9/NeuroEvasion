"""
agents/multi_discrete_agent.py — DQN agent for the Multi-Discrete action space.

MULTI-DISCRETE DQN:
    Extends DQNAgent with a dual Q-head:

        - move_head  → argmax over movement actions
        - tool_head  → argmax over tool-use actions

    Both heads are trained simultaneously via separate TD losses but share
    the same CNN backbone (gradient flows back through both heads).

TRAINING LOSSES:
    L_move = MSE( Q_move(s, move_a),  r + γ * max_a' Q_move_target(s', a') )
    L_tool = MSE( Q_tool(s, tool_a),  r + γ * max_a' Q_tool_target(s', a') )
    L_total = L_move + L_tool

    Using the same reward signal r for both heads is intentional: both
    movement and tool-use choices contribute to the same episode outcome.

CHECKPOINT COMPATIBILITY:
    get_full_state() / load_full_state() add "move_head" / "tool_head"
    keys in addition to the base "policy_net" / "target_net" keys. The
    CheckpointManager stores/loads these transparently.
"""

from __future__ import annotations

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
        - epsilon schedule and _update_epsilon()
        - replay buffer management
        - target network periodic sync
        - checkpoint save / load wrappers

    Overrides:
        - __init__    — builds dual-head networks instead of single-head
        - select_action → MultiDiscreteAction
        - train_step  → dual-loss backprop
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

        MULTI-DISCRETE TD LOSSES:
            We decode the stored action into (move_a, tool_a).
            Compute separate Bellman targets using the respective Q-heads
            of the target network, then sum the two MSE losses.

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

        # ── Target Q-values ────────────────────────────────────────────────
        with torch.no_grad():
            next_move_q, next_tool_q = self.target_net(next_states)
            # Bellman targets: r + γ * max_a' Q_target(s', a')
            move_target = rewards + self.config.gamma * next_move_q.max(dim=1)[0] * (1 - dones)
            tool_target = rewards + self.config.gamma * next_tool_q.max(dim=1)[0] * (1 - dones)

        # ── Combined loss + backprop ───────────────────────────────────────
        loss_move = nn.functional.mse_loss(move_q, move_target)
        loss_tool = nn.functional.mse_loss(tool_q, tool_target)
        loss = loss_move + loss_tool

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

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

    def sync_target_network(self) -> None:
        """Copy policy network weights to target network."""
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
