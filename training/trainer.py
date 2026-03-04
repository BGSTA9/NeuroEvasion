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

MULTI-DISCRETE MODE:
    When config.multi_discrete.use_multi_discrete is True, both agents
    are instantiated as MultiDiscreteDQNAgent (dual-head policy). All
    other training mechanics — checkpointing, logging, epsilon decay,
    target sync — are identical.

    Tool selection frequencies are accumulated per episode and passed to
    the logger as `tool_counts` for TensorBoard + CSV analysis.

CHECKPOINT & RESUME:
    Training is automatically resumed from the latest checkpoint if one exists.
    On Google Colab, the H100 runtime may be preempted at any time. This module
    handles that gracefully:

        1. At startup, CheckpointManager scans for the latest valid checkpoint.
        2. If found, agents and all training counters are restored exactly.
        3. SIGTERM and SIGINT signals trigger an emergency save before exit.
        4. Regular saves happen every `config.checkpoint.interval` episodes.

    Nothing is ever lost beyond the episodes since the last save.
"""

import os # os module
import sys # sys module
import time # time module
import signal # signal module
import logging # logging module
import numpy as np # numpy module

from config import Config # config module
from environment.env import NeuroEvasionEnv # environment module
from agents.dqn_agent import DQNAgent # dqn agent module
from training.logger import TrainingLogger # logger module
from training.checkpoint_manager import CheckpointManager # checkpoint manager module
from utils import set_global_seed # utils module
from game.actions import SNAKE_TOOL_LABELS, BAIT_TOOL_LABELS # game actions module

log = logging.getLogger(__name__) # logger


def _make_agents(config: Config, env: NeuroEvasionEnv) -> tuple: # make agents function
    """
    Factory: construct the correct agent type based on config.

    Returns:
        (snake_agent, bait_agent, is_multi_discrete)
    """
    md = config.multi_discrete # multi_discrete config, meaning: snake and bait use different action spaces

    if md.use_multi_discrete: # if multi_discrete is enabled
        from agents.multi_discrete_agent import MultiDiscreteDQNAgent # import MultiDiscreteDQNAgent

        snake_agent = MultiDiscreteDQNAgent(
            in_channels       = env.obs_channels, # number of channels in the observation space
            grid_size         = config.game.grid_size, # grid size of the game
            num_move_actions  = env.snake_num_actions, # number of move actions
            num_tool_actions  = env.snake_num_tool_actions, # number of tool actions
            config            = config.agent, # agent config
            device            = config.training.device, # device to run the agent on
            use_dueling       = md.use_dueling, # use dueling DQN
        )
        bait_agent = MultiDiscreteDQNAgent(
            in_channels       = env.obs_channels, # number of channels in the observation space
            grid_size         = config.game.grid_size, # grid size of the game
            num_move_actions  = env.bait_num_actions, # number of move actions
            num_tool_actions  = env.bait_num_tool_actions, # number of tool actions
            config            = config.agent, # agent config
            device            = config.training.device, # device to run the agent on
            use_dueling       = md.use_dueling, # use dueling DQN
        )
        return snake_agent, bait_agent, True # return agents and is_multi_discrete flag

    else:
        snake_agent = DQNAgent(
            in_channels  = env.obs_channels, # number of channels in the observation space
            grid_size    = config.game.grid_size, # grid size of the game
            num_actions  = env.snake_num_actions, # number of actions
            config       = config.agent, # agent config
            device       = config.training.device, # device to run the agent on
        )
        bait_agent = DQNAgent(
            in_channels  = env.obs_channels, # number of channels in the observation space
            grid_size    = config.game.grid_size, # grid size of the game
            num_actions  = env.bait_num_actions, # number of actions
            config       = config.agent, # agent config
            device       = config.training.device, # device to run the agent on
        )
        return snake_agent, bait_agent, False


def train(config: Config) -> None: # main training function
    """
    Main training function.

    Creates the environment and both agents, then runs the co-evolutionary
    training loop with robust checkpoint/resume support.

    Args:
        config: Master Config dataclass.  Modify config.checkpoint.* to tune
                saving behaviour; set config.checkpoint.drive_sync_dir to a
                mounted Google Drive path for Colab persistence.
    """
    # ─────────────────────────────────────────────────────────────────────────
    #  Setup
    # ─────────────────────────────────────────────────────────────────────────
    set_global_seed(config.seed) # set global seed for reproducibility

    env = NeuroEvasionEnv(config) # create environment
    snake_agent, bait_agent, is_multi_discrete = _make_agents(config, env) # create agents

    mode_label = "MULTI-DISCRETE" if is_multi_discrete else "SINGLE-DISCRETE" # mode label

    # ─────────────────────────────────────────────────────────────────────────
    #  Checkpoint manager
    # ─────────────────────────────────────────────────────────────────────────
    ckpt_cfg = config.checkpoint # checkpoint config
    manager = CheckpointManager( # create checkpoint manager
        checkpoint_dir = ckpt_cfg.checkpoint_dir, # checkpoint directory
        keep_last_n    = ckpt_cfg.keep_last_n, # keep last n checkpoints
        save_optimizer = ckpt_cfg.save_optimizer, # save optimizer
        atomic_write   = ckpt_cfg.atomic_write, # atomic write
        drive_sync_dir = ckpt_cfg.drive_sync_dir, # drive sync directory
    )

    # ─────────────────────────────────────────────────────────────────────────
    #  Resume detection
    # ─────────────────────────────────────────────────────────────────────────
    start_episode = 1
    snake_wins    = 0
    total_games   = 0
    log_dir       = f"logs/run_{int(time.time())}"    # default: fresh run

    ckpt = manager.load_latest()
    if ckpt is not None:
        # ── Restore agents ───────────────────────────────────────────────────
        snake_agent.load_full_state(ckpt["snake_state"])
        bait_agent.load_full_state(ckpt["bait_state"])

        # ── Restore training counters ────────────────────────────────────────
        meta          = ckpt["training_meta"]
        start_episode = meta["episode"] + 1
        snake_wins    = meta.get("snake_wins", 0)
        total_games   = meta.get("total_games", 0)
        log_dir       = meta.get("log_dir", log_dir)

        print(f"\n{'=' * 60}")
        print(f"  ▶️  RESUMING from episode {meta['episode']:,d}")
        print(f"  Mode:        {mode_label}")
        print(f"  Checkpoint:  {meta.get('subdir', 'unknown')}")
        print(f"  Log dir:     {log_dir}")
        print(f"  ε (epsilon): {snake_agent.epsilon:.4f}")
        print(f"  Steps done:  {snake_agent.steps_done:,d}")
        print(f"{'=' * 60}\n")

        logger = TrainingLogger.resume_from(log_dir, start_episode=start_episode)
    else:
        print(f"\n🧠 NeuroEvasion — Co-Evolutionary Training  [FRESH START]")
        print(f"{'=' * 60}")
        logger = TrainingLogger(log_dir)

    # ─────────────────────────────────────────────────────────────────────────
    #  Print training header
    # ─────────────────────────────────────────────────────────────────────────
    grid_size = config.game.grid_size
    print(f"  Mode:           {mode_label}")
    print(f"  Episodes:       {config.training.num_episodes:,} "
          f"(starting from {start_episode:,})")
    print(f"  Grid:           {grid_size}×{grid_size}")
    print(f"  Device:         {config.training.device}")
    print(f"  Snake move acts:{env.snake_num_actions}")
    print(f"  Bait move acts: {env.bait_num_actions}")
    if is_multi_discrete:
        print(f"  Snake tools:    {env.snake_num_tool_actions}  "
              f"({', '.join(SNAKE_TOOL_LABELS.values())})")
        print(f"  Bait tools:     {env.bait_num_tool_actions}  "
              f"({', '.join(BAIT_TOOL_LABELS.values())})")
    print(f"  Batch size:     {config.agent.batch_size}")
    print(f"  LR:             {config.agent.learning_rate}")
    print(f"  Seed:           {config.seed}")
    print(f"  Save every:     {ckpt_cfg.interval:,} episodes  "
          f"(keep last {ckpt_cfg.keep_last_n})")
    if ckpt_cfg.drive_sync_dir:
        print(f"  Drive sync:     {ckpt_cfg.drive_sync_dir}")
    print(f"{'=' * 60}\n")

    # ─────────────────────────────────────────────────────────────────────────
    #  SIGTERM / SIGINT handler — critical for Colab H100 preemption
    # ─────────────────────────────────────────────────────────────────────────
    _state = {"current_episode": start_episode, "saved": False}

    def _emergency_save(signum, frame):
        ep = _state["current_episode"]
        print(f"\n⚠️  Signal {signum} received at episode {ep:,d} — "
              f"saving emergency checkpoint …")
        try:
            manager.save_emergency(
                episode=ep,
                snake_agent=snake_agent,
                bait_agent=bait_agent,
                training_meta={
                    "log_dir":     log_dir,
                    "snake_wins":  snake_wins,
                    "total_games": total_games,
                },
            )
            logger.close()
            print(f"  ✅ Emergency checkpoint saved.  "
                  f"Resume with: `python main.py train`")
        except Exception as exc:
            print(f"  ❌ Emergency save failed: {exc}")
        finally:
            _state["saved"] = True
            sys.exit(0)

    signal.signal(signal.SIGTERM, _emergency_save)
    signal.signal(signal.SIGINT,  _emergency_save)

    # ─────────────────────────────────────────────────────────────────────────
    #  Training loop
    # ─────────────────────────────────────────────────────────────────────────
    for episode in range(start_episode, config.training.num_episodes + 1):
        _state["current_episode"] = episode

        snake_obs, bait_obs = env.reset()
        ep_snake_reward = 0.0
        ep_bait_reward  = 0.0
        ep_snake_loss   = 0.0
        ep_bait_loss    = 0.0
        loss_count      = 0
        step            = 0

        # Tool frequency counters (only used in multi-discrete mode)
        snake_tool_counts: dict[str, int] = {n: 0 for n in SNAKE_TOOL_LABELS.values()}
        bait_tool_counts:  dict[str, int] = {n: 0 for n in BAIT_TOOL_LABELS.values()}

        for step in range(config.game.max_steps):
            # Both agents select actions
            snake_action = snake_agent.select_action(snake_obs)
            bait_action  = bait_agent.select_action(bait_obs)

            # Environment step
            (snake_obs_next, bait_obs_next,
             snake_reward, bait_reward, done, info) = env.step(snake_action, bait_action)

            # Track tool usage (multi-discrete mode)
            if is_multi_discrete:
                s_tool = info.get("snake_tool", "NONE")
                b_tool = info.get("bait_tool",  "NONE")
                snake_tool_counts[s_tool] = snake_tool_counts.get(s_tool, 0) + 1
                bait_tool_counts[b_tool]  = bait_tool_counts.get(b_tool, 0) + 1

            # Store experiences
            snake_agent.store_transition(
                snake_obs, snake_action, snake_reward, snake_obs_next, done)
            bait_agent.store_transition(
                bait_obs,  bait_action,  bait_reward,  bait_obs_next,  done)

            # Train both agents
            s_loss = snake_agent.train_step()
            b_loss = bait_agent.train_step()

            if s_loss is not None:
                ep_snake_loss += s_loss
                ep_bait_loss  += b_loss if b_loss else 0
                loss_count    += 1

            ep_snake_reward += snake_reward
            ep_bait_reward  += bait_reward

            snake_obs = snake_obs_next
            bait_obs  = bait_obs_next

            if done:
                break

        # ── Post-Episode ──────────────────────────────────────────────────────
        total_games += 1
        if info.get("event") == "capture":
            snake_wins += 1

        avg_s_loss = ep_snake_loss / max(loss_count, 1)
        avg_b_loss = ep_bait_loss  / max(loss_count, 1)

        # Sync target networks periodically
        if episode % config.agent.target_sync_interval == 0:
            snake_agent.sync_target_network()
            bait_agent.sync_target_network()

        # Build optional tool_counts dict for logger
        tool_counts = (
            {"snake": snake_tool_counts, "bait": bait_tool_counts}
            if is_multi_discrete else None
        )

        # Log metrics
        logger.log_episode(
            episode      = episode,
            snake_reward = ep_snake_reward,
            bait_reward  = ep_bait_reward,
            snake_loss   = avg_s_loss,
            bait_loss    = avg_b_loss,
            winner       = info.get("event", "unknown"),
            steps        = step + 1,
            epsilon      = snake_agent.epsilon,
            tool_counts  = tool_counts,
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
            if is_multi_discrete:
                # Compact tool selection summary
                s_top = max(snake_tool_counts, key=snake_tool_counts.get)
                b_top = max(bait_tool_counts,  key=bait_tool_counts.get)
                print(
                    f"         🛠 Snake top-tool: {s_top}={snake_tool_counts[s_top]} | "
                    f"Bait top-tool: {b_top}={bait_tool_counts[b_top]}"
                )
            # Reset rolling counters
            snake_wins  = 0
            total_games = 0

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if episode % ckpt_cfg.interval == 0:
            save_path = manager.save(
                episode       = episode,
                snake_agent   = snake_agent,
                bait_agent    = bait_agent,
                training_meta = {
                    "log_dir":     log_dir,
                    "snake_wins":  snake_wins,
                    "total_games": total_games,
                },
            )
            print(f"  💾 Checkpoint saved → {save_path} (episode {episode:,d})")

    # ─────────────────────────────────────────────────────────────────────────
    #  Final save
    # ─────────────────────────────────────────────────────────────────────────
    final_path = manager.save(
        episode       = config.training.num_episodes,
        snake_agent   = snake_agent,
        bait_agent    = bait_agent,
        training_meta = {
            "log_dir":     log_dir,
            "snake_wins":  snake_wins,
            "total_games": total_games,
            "final":       True,
        },
    )
    # Also write legacy flat files for the demo/evaluate commands
    snake_agent.save(f"{ckpt_cfg.checkpoint_dir}/snake_final.pt")
    bait_agent.save(f"{ckpt_cfg.checkpoint_dir}/bait_final.pt")

    logger.close()
    print(f"\n✅ Training complete!  Final checkpoint → {final_path}")
    print(f"   Legacy flat files  → {ckpt_cfg.checkpoint_dir}/snake_final.pt")
