"""
environment/env.py — Gymnasium-compatible environment wrapper.

This wraps the GameEngine + StateEncoder + FrameStacker into a clean
interface that the training loop and agents interact with.

THE GYM API PATTERN:
    env.reset()  → observations
    env.step(actions) → observations, rewards, done, info

    This is the standard interface used across the RL community
    (OpenAI Gym, Gymnasium, DeepMind Lab, etc.). Learning this
    pattern transfers to any RL project.
"""

import numpy as np
from game.engine import GameEngine
from environment.state_encoder import StateEncoder
from environment.frame_stacker import FrameStacker
from config import Config


class NeuroEvasionEnv:
    """
    Multi-agent environment for the NeuroEvasion game.

    Provides separate observations for snake and bait agents.

    Usage:
        env = NeuroEvasionEnv(config)
        snake_obs, bait_obs = env.reset()

        while not done:
            snake_obs, bait_obs, snake_r, bait_r, done, info = env.step(s_act, b_act)
    """

    def __init__(self, config: Config):
        self.config = config
        self.engine = GameEngine(config.game, config.rewards)
        self.encoder = StateEncoder(config.game.grid_size)

        obs_shape = (4, config.game.grid_size, config.game.grid_size)
        self.snake_stacker = FrameStacker(config.agent.frame_stack, obs_shape)
        self.bait_stacker = FrameStacker(config.agent.frame_stack, obs_shape)

        # Action and observation space info
        self.snake_num_actions = 4   # UP, DOWN, LEFT, RIGHT
        self.bait_num_actions = 5 if config.game.bait_can_stay else 4
        self.obs_channels = 4 * config.agent.frame_stack
        self.obs_height = config.game.grid_size
        self.obs_width = config.game.grid_size

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
        bait_stacked = self.bait_stacker.reset(bait_obs)

        return snake_stacked, bait_stacked

    def step(self, snake_action: int, bait_action: int) -> tuple:
        """
        Execute one environment step.

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
            bait_obs = np.zeros_like(self.bait_stacker.get_stacked()[:4])

        snake_stacked = self.snake_stacker.push(snake_obs)
        bait_stacked = self.bait_stacker.push(bait_obs)

        return snake_stacked, bait_stacked, snake_reward, bait_reward, done, info
