"""
agents/networks.py — Neural network architectures for DQN agents.

MULTI-DISCRETE EXTENSION:
    MultiDiscreteNetwork and MultiDiscreteDuelingNetwork implement a
    shared-backbone / dual-head architecture:

        Shared CNN (feature extractor)
            ├── move_head → Q(s, move_a)   [num_move_actions outputs]
            └── tool_head → Q(s, tool_a)   [num_tool_actions outputs]

    WHY A SHARED BACKBONE?
        Both action heads need to understand the same spatial features:
        where the snake is, where the bait is, how close are the walls.
        Sharing the CNN means we train those features once, not twice,
        reducing parameters and computation while improving sample efficiency.

    The Dueling variant further decomposes each head into V(s) + A(s,a)
    for better value estimation in sparse-reward conditions.

ARCHITECTURE OVERVIEW:
    We use a Convolutional Neural Network (CNN) because our observations
    are 2D grids (like images). The CNN learns spatial patterns:

    Conv layers → extract features ("snake is nearby", "wall ahead")
    FC layers   → combine features into action-value estimates

WHAT IS A Q-VALUE?
    Q(s, a) estimates the total future reward if we take action 'a'
    in state 's' and then follow our policy. The agent picks the
    action with the highest Q-value: argmax_a Q(s, a).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Deep Q-Network with convolutional feature extraction.

    Architecture:
        3 × Conv2d layers (feature extraction)
        → Flatten
        → 2 × Linear layers (decision making)
        → Q-value output (one per action)

    Args:
        in_channels: number of input channels (e.g., 12 for 3-frame stack × 4 channels)
        grid_size: height/width of the grid (e.g., 20)
        num_actions: number of possible actions (e.g., 4 or 5)
    """

    def __init__(self, in_channels: int, grid_size: int, num_actions: int):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Layer 1: Detect basic patterns (edges, positions)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 2: Combine basic patterns into complex features
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 3: Higher-level strategic features
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Calculate flattened size after convolutions
        # With padding=1 and stride=1, spatial dimensions are preserved
        flat_size = 64 * grid_size * grid_size

        self.fc_layers = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),  # Output: Q(s,a) for each action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: observation → Q-values.

        Args:
            x: observation tensor of shape (batch, channels, H, W)

        Returns:
            Q-values of shape (batch, num_actions)
        """
        features = self.conv_layers(x)
        flat = features.view(features.size(0), -1)  # Flatten
        q_values = self.fc_layers(flat)
        return q_values


class DuelingDQN(nn.Module):
    """
    Dueling DQN — separates state value from action advantage.

    KEY INSIGHT:
        Sometimes it's important to know the VALUE of a state regardless
        of which action we take. For example, if we're about to die,
        ALL actions have low value. The Dueling architecture learns this
        decomposition explicitly:

        Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))

        V(s)    = "how good is this state in general?"
        A(s, a) = "how much better/worse is this specific action?"

    PAPER: Wang et al., "Dueling Network Architectures for Deep RL" (2016)
    """

    def __init__(self, in_channels: int, grid_size: int, num_actions: int):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        flat_size = 64 * grid_size * grid_size

        # Value stream: "How good is this state?"
        self.value_stream = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Advantage stream: "How much better is each action?"
        self.advantage_stream = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        flat = features.view(features.size(0), -1)

        value = self.value_stream(flat)           # (batch, 1)
        advantage = self.advantage_stream(flat)   # (batch, actions)

        # Q = V + (A - mean(A))  → ensures identifiability
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


# ──────────────────────────────────────────────────────────────────────────────
#  Multi-Discrete Network Architectures
# ──────────────────────────────────────────────────────────────────────────────

class MultiDiscreteNetwork(nn.Module):
    """
    Multi-head DQN for the Multi-Discrete action space.

    Outputs TWO independent Q-value vectors per forward pass:
        move_q:  Q(s, movement_action)   shape (batch, num_move_actions)
        tool_q:  Q(s, tool_action)       shape (batch, num_tool_actions)

    Both heads share the expensive CNN feature extractor, so total
    parameter count grows only slightly versus the single-head version.

    Args:
        in_channels:       input channels (frame_stack × 4)
        grid_size:         H=W of the observation grid
        num_move_actions:  number of discrete movement actions (4 or 5)
        num_tool_actions:  number of discrete tool actions (3 by default)
    """

    def __init__(self, in_channels: int, grid_size: int,
                 num_move_actions: int, num_tool_actions: int):
        super().__init__()

        # ── Shared convolutional backbone ──────────────────────────────────────
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        flat_size = 64 * grid_size * grid_size

        # ── Movement Q-head ────────────────────────────────────────────────────
        self.move_head = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_move_actions),
        )

        # ── Tool-use Q-head ────────────────────────────────────────────────────
        self.tool_head = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_tool_actions),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: observation → (move_q, tool_q).

        Args:
            x: (batch, in_channels, H, W)

        Returns:
            move_q: (batch, num_move_actions)
            tool_q: (batch, num_tool_actions)
        """
        features = self.conv_layers(x)
        flat = features.view(features.size(0), -1)
        return self.move_head(flat), self.tool_head(flat)


class MultiDiscreteDuelingNetwork(nn.Module):
    """
    Dueling Multi-head DQN for the Multi-Discrete action space.

    Each action head uses the Dueling decomposition:
        Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))

    This is especially powerful for the tool head, where most states have
    tool=NONE as the dominant action — Dueling's V(s) term correctly
    captures that the state value is independent of which (uncommon) tool
    is chosen.

    Args:
        in_channels:       input channels (frame_stack × 4)
        grid_size:         H=W of the observation grid
        num_move_actions:  number of discrete movement actions (4 or 5)
        num_tool_actions:  number of discrete tool actions (3 by default)
    """

    def __init__(self, in_channels: int, grid_size: int,
                 num_move_actions: int, num_tool_actions: int):
        super().__init__()

        # ── Shared convolutional backbone ──────────────────────────────────────
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        flat_size = 64 * grid_size * grid_size

        # ── Movement head: Dueling streams ─────────────────────────────────────
        self.move_value = nn.Sequential(
            nn.Linear(flat_size, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.move_advantage = nn.Sequential(
            nn.Linear(flat_size, 512), nn.ReLU(), nn.Linear(512, num_move_actions)
        )

        # ── Tool head: Dueling streams ─────────────────────────────────────────
        self.tool_value = nn.Sequential(
            nn.Linear(flat_size, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        self.tool_advantage = nn.Sequential(
            nn.Linear(flat_size, 256), nn.ReLU(), nn.Linear(256, num_tool_actions)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: observation → (move_q, tool_q) with Dueling decomposition.

        Args:
            x: (batch, in_channels, H, W)

        Returns:
            move_q: (batch, num_move_actions)
            tool_q: (batch, num_tool_actions)
        """
        features = self.conv_layers(x)
        flat = features.view(features.size(0), -1)

        # Movement head
        mv = self.move_value(flat)                    # (batch, 1)
        ma = self.move_advantage(flat)               # (batch, num_move_actions)
        move_q = mv + ma - ma.mean(dim=1, keepdim=True)

        # Tool head
        tv = self.tool_value(flat)                   # (batch, 1)
        ta = self.tool_advantage(flat)               # (batch, num_tool_actions)
        tool_q = tv + ta - ta.mean(dim=1, keepdim=True)

        return move_q, tool_q
