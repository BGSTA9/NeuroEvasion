"""
evaluation/evaluator.py — Evaluate trained agents over many games.

Computes key performance metrics without rendering, for fast evaluation.
"""

import numpy as np
from config import Config
from environment.env import NeuroEvasionEnv
from agents.dqn_agent import DQNAgent


def evaluate(config: Config, snake_model_path: str,
             bait_model_path: str, num_games: int = 1000) -> dict:
    """
    Run evaluation games and compute statistics.

    Returns:
        dict with snake_win_rate, avg_survival, avg_snake_reward, etc.
    """
    env = NeuroEvasionEnv(config)
    in_channels = env.obs_channels
    grid_size = config.game.grid_size

    snake_agent = DQNAgent(in_channels, grid_size, env.snake_num_actions,
                           config.agent, device="cpu")
    bait_agent = DQNAgent(in_channels, grid_size, env.bait_num_actions,
                          config.agent, device="cpu")

    snake_agent.load(snake_model_path)
    bait_agent.load(bait_model_path)
    snake_agent.epsilon = 0.0
    bait_agent.epsilon = 0.0

    snake_wins = 0
    survival_steps = []
    snake_rewards = []
    bait_rewards = []

    print(f"Evaluating over {num_games:,d} games...")

    for game in range(num_games):
        snake_obs, bait_obs = env.reset()
        ep_snake_r = 0.0
        ep_bait_r = 0.0
        done = False
        steps = 0

        while not done:
            s_action = snake_agent.select_action(snake_obs)
            b_action = bait_agent.select_action(bait_obs)
            snake_obs, bait_obs, s_r, b_r, done, info = env.step(s_action, b_action)
            ep_snake_r += s_r
            ep_bait_r += b_r
            steps += 1

        if info.get("event") == "capture":
            snake_wins += 1

        survival_steps.append(steps)
        snake_rewards.append(ep_snake_r)
        bait_rewards.append(ep_bait_r)

        if (game + 1) % 100 == 0:
            print(f"  Evaluated {game + 1:,d}/{num_games:,d} games...")

    results = {
        "snake_win_rate": snake_wins / num_games,
        "bait_win_rate": 1 - snake_wins / num_games,
        "avg_survival_steps": np.mean(survival_steps),
        "median_survival_steps": np.median(survival_steps),
        "avg_snake_reward": np.mean(snake_rewards),
        "avg_bait_reward": np.mean(bait_rewards),
        "total_games": num_games,
    }

    print(f"\n{'=' * 50}")
    print(f"  EVALUATION RESULTS")
    print(f"{'=' * 50}")
    print(f"  Games played:          {num_games:,d}")
    print(f"  Snake win rate:        {results['snake_win_rate']:.1%}")
    print(f"  Bait win rate:         {results['bait_win_rate']:.1%}")
    print(f"  Avg survival (steps):  {results['avg_survival_steps']:.1f}")
    print(f"  Median survival:       {results['median_survival_steps']:.1f}")
    print(f"  Avg snake reward:      {results['avg_snake_reward']:.2f}")
    print(f"  Avg bait reward:       {results['avg_bait_reward']:.2f}")
    print(f"{'=' * 50}")

    return results
