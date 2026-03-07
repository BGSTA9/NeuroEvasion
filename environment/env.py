"""
environment/env.py — Gymnasium-compatible environment wrapper.

This wraps the GameEngine + StateEncoder + FrameStacker into a clean
interface that the training loop and agents interact with.

MULTI-DISCRETE EXTENSION:
    When config.multi_discrete.use_multi_discrete is True, the env:
      - exposes snake_num_tool_actions / bait_num_tool_actions
      - wraps raw integer actions into MultiDiscreteAction before calling engine.step()

    When False (default), behaviour is 100% identical to the original.

THE GYM API PATTERN:
    env.reset()  → observations
    env.step(actions) → observations, rewards, done, info
    env.render() → (H, W, 3) uint8 numpy array (RGB image for video logging)
    env.record_eval_episode(snake, bait) → list of frames (numpy arrays)

    This is the standard interface used across the RL community
    (OpenAI Gym, Gymnasium, DeepMind Lab, etc.). Learning this
    pattern transfers to any RL project.
"""

import numpy as np
from game.engine import GameEngine
from game.actions import MultiDiscreteAction
from environment.state_encoder import StateEncoder
from environment.frame_stacker import FrameStacker
from config import Config


class NeuroEvasionEnv:
    """
    Multi-agent environment for the NeuroEvasion game.

    Provides separate observations for snake and bait agents.

    Usage (single-discrete, legacy):
        env = NeuroEvasionEnv(config)
        snake_obs, bait_obs = env.reset()
        while not done:
            snake_obs, bait_obs, snake_r, bait_r, done, info = env.step(s_act, b_act)

    Usage (multi-discrete):
        config.multi_discrete.use_multi_discrete = True
        env = NeuroEvasionEnv(config)
        # step() accepts either plain ints OR MultiDiscreteAction instances
        while not done:
            snake_obs, bait_obs, snake_r, bait_r, done, info = env.step(
                MultiDiscreteAction(move=2, tool=1),
                MultiDiscreteAction(move=0, tool=0),
            )
    """

    def __init__(self, config: Config):
        self.config = config
        self.engine = GameEngine(config.game, config.rewards)
        self.encoder = StateEncoder(config.game.grid_size)

        obs_shape = (4, config.game.grid_size, config.game.grid_size)
        self.snake_stacker = FrameStacker(config.agent.frame_stack, obs_shape)
        self.bait_stacker  = FrameStacker(config.agent.frame_stack, obs_shape)

        # ── Action space info ──────────────────────────────────────────────────
        self.snake_num_actions      = 4   # UP, DOWN, LEFT, RIGHT
        self.bait_num_actions       = 5 if config.game.bait_can_stay else 4
        self.obs_channels           = 4 * config.agent.frame_stack
        self.obs_height             = config.game.grid_size
        self.obs_width              = config.game.grid_size

        # Multi-discrete extra attributes (always set; ignored in legacy mode)
        md = config.multi_discrete
        self.use_multi_discrete     = md.use_multi_discrete
        self.snake_num_tool_actions = md.snake_num_tool_actions
        self.bait_num_tool_actions  = md.bait_num_tool_actions

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Reset the environment for a new episode.

        Returns:
            (snake_observation, bait_observation) — stacked frame tensors
        """
        self.engine.reset()
        state = self.engine.get_state()

        snake_obs = self.encoder.encode_for_snake(
            self.engine.grid, state["snake_body"],
            state["snake_head"], state["bait_pos"]
        )
        bait_obs = self.encoder.encode_for_bait(
            self.engine.grid, state["snake_body"],
            state["snake_head"], state["bait_pos"]
        )

        snake_stacked = self.snake_stacker.reset(snake_obs)
        bait_stacked  = self.bait_stacker.reset(bait_obs)

        return snake_stacked, bait_stacked

    def step(
        self,
        snake_action: int | MultiDiscreteAction,
        bait_action:  int | MultiDiscreteAction,
    ) -> tuple:
        """
        Execute one environment step.

        Accepts plain int actions (single-discrete, legacy) OR
        MultiDiscreteAction instances (multi-discrete mode) — the engine
        handles both transparently.

        Returns:
            (snake_obs, bait_obs, snake_reward, bait_reward, done, info)
        """
        snake_reward, bait_reward, done, info = self.engine.step(snake_action, bait_action)

        state = self.engine.get_state()

        if not done:
            snake_obs = self.encoder.encode_for_snake(
                self.engine.grid, state["snake_body"],
                state["snake_head"], state["bait_pos"]
            )
            bait_obs = self.encoder.encode_for_bait(
                self.engine.grid, state["snake_body"],
                state["snake_head"], state["bait_pos"]
            )
        else:
            # Terminal state — return zeros (won't be used for learning)
            snake_obs = np.zeros_like(self.snake_stacker.get_stacked()[:4])
            bait_obs  = np.zeros_like(self.bait_stacker.get_stacked()[:4])

        snake_stacked = self.snake_stacker.push(snake_obs)
        bait_stacked  = self.bait_stacker.push(bait_obs)

        return snake_stacked, bait_stacked, snake_reward, bait_reward, done, info

    def render(self, cell_size: int = 16) -> np.ndarray:
        """
        Render the current game state as an RGB numpy array.
        This runs completely headless (no Pygame required), perfect for Colab.
        
        Args:
            cell_size: Size of each grid cell in pixels.
            
        Returns:
            A (H, W, 3) uint8 numpy array image.
        """
        h = self.config.game.grid_size * cell_size
        w = self.config.game.grid_size * cell_size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Colors (RGB)
        C_WALL  = (40, 44, 52)
        C_EMPTY = (15, 17, 26)
        C_SNAKE = (97, 175, 239)
        C_HEAD  = (86, 182, 194)
        C_BAIT  = (224, 108, 117)

        grid = self.engine.grid.cells
        
        for r in range(self.config.game.grid_size):
            for c in range(self.config.game.grid_size):
                val = grid[r, c]
                y1, y2 = r * cell_size, (r + 1) * cell_size
                x1, x2 = c * cell_size, (c + 1) * cell_size
                
                if val == 1:   # Wall
                    img[y1:y2, x1:x2] = C_WALL
                elif val == 2: # Snake Body
                    img[y1:y2, x1:x2] = C_SNAKE
                elif val == 3: # Snake Head
                    img[y1:y2, x1:x2] = C_HEAD
                elif val == 4: # Bait
                    img[y1:y2, x1:x2] = C_BAIT
                else:          # Empty / Unknown
                    img[y1:y2, x1:x2] = C_EMPTY
                    
        return img

    def record_eval_episode(self, snake_agent, bait_agent, max_steps: int = 200, cell_size: int = 16) -> list[np.ndarray]:
        """
        Run one greedy (epsilon=0) episode and record the frames.
        
        Returns:
            List of (H, W, 3) uint8 numpy array frames for video generation.
        """
        s_obs, b_obs = self.reset()
        frames = [self.render(cell_size)]
        
        # Temporarily force greedy actions
        s_eps_bak = snake_agent.epsilon
        b_eps_bak = bait_agent.epsilon
        snake_agent.epsilon = 0.0
        bait_agent.epsilon = 0.0
        
        for _ in range(max_steps):
            s_act = snake_agent.select_action(s_obs)
            b_act = bait_agent.select_action(b_obs)
            
            s_obs, b_obs, _, _, done, _ = self.step(s_act, b_act)
            frames.append(self.render(cell_size))
            
            if done:
                break
                
        # Restore exploration rates
        snake_agent.epsilon = s_eps_bak
        bait_agent.epsilon = b_eps_bak
        
        return frames
