"""
training/trainer.py — Main training loop for co-evolutionary learning.

THE CO-EVOLUTION PROCESS:
    1. Both agents start with random policies (high ε)
    2. They play thousands of games against each other
    3. Each agent improves against the CURRENT version of the other
    4. As one gets better, it forces the other to adapt
    5. This creates an arms race of increasingly sophisticated strategies

EMERGENT BEHAVIORS YOU MAY OBSERVE:
    - Snake learns to "corner" the bait against walls
    - Bait learns to run along walls to maximize escape routes
    - Snake learns to "cut off" the bait's escape paths
    - Bait learns to fake one direction then go another
"""

import os
import time
import numpy as np
from config import Config
from environment.env import NeuroEvasionEnv
from agents.dqn_agent import DQNAgent
from training.logger import TrainingLogger
from utils import set_global_seed


def train(config: Config) -> None:
    """
    Main training function.

    Creates the environment and both agents, then runs the
    co-evolutionary training loop.
    """
    # --- Setup ---
    set_global_seed(config.seed)

    env = NeuroEvasionEnv(config)

    in_channels = env.obs_channels
    grid_size = config.game.grid_size

    snake_agent = DQNAgent(
        in_channels=in_channels,
        grid_size=grid_size,
        num_actions=env.snake_num_actions,
        config=config.agent,
        device=config.training.device,
    )

    bait_agent = DQNAgent(
        in_channels=in_channels,
        grid_size=grid_size,
        num_actions=env.bait_num_actions,
        config=config.agent,
        device=config.training.device,
    )

    logger = TrainingLogger(f"logs/run_{int(time.time())}")

    print(f"🧠 NeuroEvasion — Co-Evolutionary Training")
    print(f"{'=' * 60}")
    print(f"  Episodes:     {config.training.num_episodes:,}")
    print(f"  Grid:         {grid_size}×{grid_size}")
    print(f"  Device:       {config.training.device}")
    print(f"  Snake acts:   {env.snake_num_actions}")
    print(f"  Bait acts:    {env.bait_num_actions}")
    print(f"  Batch size:   {config.agent.batch_size}")
    print(f"  LR:           {config.agent.learning_rate}")
    print(f"  Seed:         {config.seed}")
    print(f"{'=' * 60}\n")

    # --- Tracking ---
    snake_wins = 0
    total_games = 0

    # --- Training Loop ---
    for episode in range(1, config.training.num_episodes + 1):
        snake_obs, bait_obs = env.reset()

        ep_snake_reward = 0.0
        ep_bait_reward = 0.0
        ep_snake_loss = 0.0
        ep_bait_loss = 0.0
        loss_count = 0
        step = 0

        for step in range(config.game.max_steps):
            # Both agents select actions
            snake_action = snake_agent.select_action(snake_obs)
            bait_action = bait_agent.select_action(bait_obs)

            # Environment step
            (snake_obs_next, bait_obs_next,
             snake_reward, bait_reward, done, info) = env.step(snake_action, bait_action)

            # Store experiences
            snake_agent.store_transition(
                snake_obs, snake_action, snake_reward, snake_obs_next, done)
            bait_agent.store_transition(
                bait_obs, bait_action, bait_reward, bait_obs_next, done)

            # Train both agents
            s_loss = snake_agent.train_step()
            b_loss = bait_agent.train_step()

            if s_loss is not None:
                ep_snake_loss += s_loss
                ep_bait_loss += b_loss if b_loss else 0
                loss_count += 1

            ep_snake_reward += snake_reward
            ep_bait_reward += bait_reward

            snake_obs = snake_obs_next
            bait_obs = bait_obs_next

            if done:
                break

        # --- Post-Episode ---
        total_games += 1
        if info.get("event") == "capture":
            snake_wins += 1

        avg_s_loss = ep_snake_loss / max(loss_count, 1)
        avg_b_loss = ep_bait_loss / max(loss_count, 1)

        # Sync target networks periodically
        if episode % config.agent.target_sync_interval == 0:
            snake_agent.sync_target_network()
            bait_agent.sync_target_network()

        # Log metrics
        logger.log_episode(
            episode=episode,
            snake_reward=ep_snake_reward,
            bait_reward=ep_bait_reward,
            snake_loss=avg_s_loss,
            bait_loss=avg_b_loss,
            winner=info.get("event", "unknown"),
            steps=step + 1,
            epsilon=snake_agent.epsilon,
        )

        # Print progress
        if episode % config.training.log_interval == 0:
            win_rate = snake_wins / max(total_games, 1) * 100
            print(
                f"Ep {episode:>7,d} | "
                f"ε={snake_agent.epsilon:.3f} | "
                f"🐍 R={ep_snake_reward:>7.2f} | "
                f"🎯 R={ep_bait_reward:>7.2f} | "
                f"Win%={win_rate:>5.1f}% | "
                f"Steps={step+1:>3d} | "
                f"Loss={avg_s_loss:.4f}"
            )
            # Reset rolling counters
            snake_wins = 0
            total_games = 0

        # Save checkpoints
        if episode % config.training.checkpoint_interval == 0:
            snake_agent.save(f"checkpoints/snake_ep{episode}.pt")
            bait_agent.save(f"checkpoints/bait_ep{episode}.pt")
            print(f"  💾 Checkpoints saved at episode {episode:,d}")

    # --- Final Save ---
    snake_agent.save("checkpoints/snake_final.pt")
    bait_agent.save("checkpoints/bait_final.pt")
    logger.close()
    print(f"\n✅ Training complete! Final models saved to checkpoints/")
