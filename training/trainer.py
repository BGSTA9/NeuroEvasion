"""
training/trainer.py — Main training loop for co-evolutionary learning.

THE CO-EVOLUTION PROCESS:
    1. Both agents start with random policies (high ε)
    2. They play thousands of games against each other
    3. Each agent improves against the CURRENT version of the other
    4. As one gets better, it forces the other to adapt
    5. This creates an arms race of increasingly sophisticated strategies

COEVOLUTIONARY FIXES (v2):
    - Soft target update (Polyak τ) replaces hard sync every N episodes.
    - Cyclic ε-greedy exploration prevents policy crystallisation.
    - Historical opponent pool (30% of episodes) stabilises coevolution.
    - Running reward normalisation keeps gradient magnitudes in check.
    - Reward clipping prevents extreme TD-error spikes.
    - Double DQN (in dqn_agent.py) corrects Q-value overestimation.
    - GroupNorm (in networks.py) replaces BatchNorm for non-stationary robustness.

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

import os
import sys
import time
import signal
import logging
import numpy as np

try:
    import wandb
    _WANDB_AVAILABLE = wandb.run is not None  # True only if wandb.init() was already called
except ImportError:
    _WANDB_AVAILABLE = False

from config import Config
from environment.env import NeuroEvasionEnv
from agents.dqn_agent import DQNAgent
from training.logger import TrainingLogger
from training.checkpoint_manager import CheckpointManager
from training.reward_normalizer import RewardNormalizer
from training.opponent_pool import OpponentPool
from utils import set_global_seed
from game.actions import SNAKE_TOOL_LABELS, BAIT_TOOL_LABELS

log = logging.getLogger(__name__)


def _wandb_log(metrics: dict) -> None:
    """Safe wandb.log wrapper — silently skips if wandb is not initialised."""
    if _WANDB_AVAILABLE:
        try:
            wandb.log(metrics)
        except Exception:
            pass  # never let wandb crash the training loop


def _make_agents(config: Config, env: NeuroEvasionEnv) -> tuple:
    """
    Factory: construct the correct agent type based on config.

    Returns:
        (snake_agent, bait_agent, is_multi_discrete)
    """
    md = config.multi_discrete

    if md.use_multi_discrete:
        from agents.multi_discrete_agent import MultiDiscreteDQNAgent

        snake_agent = MultiDiscreteDQNAgent(
            in_channels       = env.obs_channels,
            grid_size         = config.game.grid_size,
            num_move_actions  = env.snake_num_actions,
            num_tool_actions  = env.snake_num_tool_actions,
            config            = config.agent,
            device            = config.training.device,
            use_dueling       = md.use_dueling,
        )
        bait_agent = MultiDiscreteDQNAgent(
            in_channels       = env.obs_channels,
            grid_size         = config.game.grid_size,
            num_move_actions  = env.bait_num_actions,
            num_tool_actions  = env.bait_num_tool_actions,
            config            = config.agent,
            device            = config.training.device,
            use_dueling       = md.use_dueling,
        )
        return snake_agent, bait_agent, True

    else:
        snake_agent = DQNAgent(
            in_channels  = env.obs_channels,
            grid_size    = config.game.grid_size,
            num_actions  = env.snake_num_actions,
            config       = config.agent,
            device       = config.training.device,
        )
        bait_agent = DQNAgent(
            in_channels  = env.obs_channels,
            grid_size    = config.game.grid_size,
            num_actions  = env.bait_num_actions,
            config       = config.agent,
            device       = config.training.device,
        )
        return snake_agent, bait_agent, False


def train(config: Config) -> None:
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
    set_global_seed(config.seed)

    env = NeuroEvasionEnv(config)
    snake_agent, bait_agent, is_multi_discrete = _make_agents(config, env)

    mode_label = "MULTI-DISCRETE" if is_multi_discrete else "SINGLE-DISCRETE"

    # ─────────────────────────────────────────────────────────────────────────
    #  Coevolutionary stabilisers
    # ─────────────────────────────────────────────────────────────────────────
    coevo = config.coevolution

    snake_reward_norm = RewardNormalizer(clip=coevo.reward_clip)
    bait_reward_norm  = RewardNormalizer(clip=coevo.reward_clip)

    snake_opponent_pool = OpponentPool(
        pool_size=coevo.pool_size, current_prob=coevo.pool_current_prob)
    bait_opponent_pool = OpponentPool(
        pool_size=coevo.pool_size, current_prob=coevo.pool_current_prob)

    # ─────────────────────────────────────────────────────────────────────────
    #  Checkpoint manager
    # ─────────────────────────────────────────────────────────────────────────
    ckpt_cfg = config.checkpoint
    manager = CheckpointManager(
        checkpoint_dir = ckpt_cfg.checkpoint_dir,
        keep_last_n    = ckpt_cfg.keep_last_n,
        save_optimizer = ckpt_cfg.save_optimizer,
        atomic_write   = ckpt_cfg.atomic_write,
        drive_sync_dir = ckpt_cfg.drive_sync_dir,
    )

    # ─────────────────────────────────────────────────────────────────────────
    #  Resume detection
    # ─────────────────────────────────────────────────────────────────────────
    start_episode = 1
    snake_wins    = 0
    total_games   = 0
    log_dir       = f"logs/run_{int(time.time())}"

    ckpt = manager.load_latest()
    if ckpt is not None:
        snake_agent.load_full_state(ckpt["snake_state"])
        bait_agent.load_full_state(ckpt["bait_state"])

        meta          = ckpt["training_meta"]
        start_episode = meta["episode"] + 1
        snake_wins    = meta.get("snake_wins", 0)
        total_games   = meta.get("total_games", 0)
        log_dir       = meta.get("log_dir", log_dir)

        # Restore coevolutionary state if present
        if "snake_reward_norm" in meta:
            snake_reward_norm.load_state(meta["snake_reward_norm"])
        if "bait_reward_norm" in meta:
            bait_reward_norm.load_state(meta["bait_reward_norm"])
        if "snake_opponent_pool" in meta:
            snake_opponent_pool.load_state(meta["snake_opponent_pool"])
        if "bait_opponent_pool" in meta:
            bait_opponent_pool.load_state(meta["bait_opponent_pool"])

        print(f"\n{'=' * 60}")
        print(f"  ▶️  RESUMING from episode {meta['episode']:,d}")
        print(f"  Mode:        {mode_label}")
        print(f"  Checkpoint:  {meta.get('subdir', 'unknown')}")
        print(f"  Log dir:     {log_dir}")
        print(f"  ε (epsilon): {snake_agent.epsilon:.4f}")
        print(f"  Steps done:  {snake_agent.steps_done:,d}")
        print(f"  Opponent pool: Snake={len(snake_opponent_pool)}, "
              f"Bait={len(bait_opponent_pool)}")
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
    print(f"  Double DQN:     {config.agent.use_double_dqn}")
    print(f"  Soft tau:       {config.agent.target_update_tau}")
    print(f"  Cyclic ε:       {config.agent.epsilon_cycle}")
    print(f"  Opponent pool:  {coevo.use_opponent_pool} "
          f"(size={coevo.pool_size}, current%={coevo.pool_current_prob:.0%})")
    print(f"  Reward norm:    {coevo.reward_normalize}")
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

    def _build_training_meta() -> dict:
        """Build the full training metadata dict for checkpointing."""
        return {
            "log_dir":             log_dir,
            "snake_wins":          snake_wins,
            "total_games":         total_games,
            "snake_reward_norm":   snake_reward_norm.get_state(),
            "bait_reward_norm":    bait_reward_norm.get_state(),
            "snake_opponent_pool": snake_opponent_pool.get_state(),
            "bait_opponent_pool":  bait_opponent_pool.get_state(),
        }

    def _emergency_save(signum, frame):
        ep = _state["current_episode"]
        print(f"\n⚠️  Signal {signum} received at episode {ep:,d} — "
              f"saving emergency checkpoint …")
        try:
            manager.save_emergency(
                episode=ep,
                snake_agent=snake_agent,
                bait_agent=bait_agent,
                training_meta=_build_training_meta(),
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
    
    _last_log_time = time.time()
    
    # ─────────────────────────────────────────────────────────────────────────
    #  Training loop
    # ─────────────────────────────────────────────────────────────────────────
    for episode in range(start_episode, config.training.num_episodes + 1):
        _state["current_episode"] = episode

        # ── Remote Control & Emergency Stop ───────────────────────────────────
        manual_stop_triggered = False
        
        # 1. Check W&B config overrides for manual stop and LR
        if _WANDB_AVAILABLE:
            if wandb.config.get("manual_stop", False):
                print(f"\n🛑 Manual Stop Triggered via W&B at episode {episode:,}!")
                manual_stop_triggered = True
                
            new_lr = wandb.config.get("learning_rate")
            current_lr = snake_agent.optimizer.param_groups[0]["lr"]
            if new_lr and abs(new_lr - current_lr) / current_lr > 0.01:  # only act on >1% change
                print(f"\n📉 Remote Control: Changing learning rate to {new_lr:.2e}")
                for param_group in snake_agent.optimizer.param_groups:
                    param_group['lr'] = new_lr
                for param_group in bait_agent.optimizer.param_groups:
                    param_group['lr'] = new_lr
                    
        # 2. Check for physical STOP_TRAINING.txt block in Drive
        if ckpt_cfg.drive_sync_dir:
            stop_file = os.path.join(ckpt_cfg.drive_sync_dir, "STOP_TRAINING.txt")
            if os.path.exists(stop_file):
                print(f"\\n🛑 Manual Stop Triggered via STOP_TRAINING.txt at episode {episode:,}!")
                manual_stop_triggered = True
                
        if manual_stop_triggered:
            save_path = manager.save(
                episode=episode,
                snake_agent=snake_agent,
                bait_agent=bait_agent,
                training_meta=_build_training_meta(),
            )
            print(f"  💾 Emergency final checkpoint saved → {save_path}")
            break

        snake_obs, bait_obs = env.reset()
        ep_snake_reward = 0.0
        ep_bait_reward  = 0.0
        ep_snake_loss   = 0.0
        ep_bait_loss    = 0.0
        loss_count      = 0
        step            = 0

        snake_tool_counts: dict[str, int] = {n: 0 for n in SNAKE_TOOL_LABELS.values()}
        bait_tool_counts:  dict[str, int] = {n: 0 for n in BAIT_TOOL_LABELS.values()}

        # ── Opponent pool: occasionally play against a historical snapshot ──
        using_historical_snake = False
        using_historical_bait  = False
        original_snake_policy = None
        original_bait_policy  = None

        if coevo.use_opponent_pool:
            # 30% chance: load a historical bait for the snake to play against
            if bait_opponent_pool.should_use_historical():
                using_historical_bait = True
                original_bait_policy = {k: v.clone() for k, v in
                                        bait_agent.policy_net.state_dict().items()}
                bait_agent.policy_net.load_state_dict(
                    bait_opponent_pool.get_random_snapshot()
                )

            # 30% chance: load a historical snake for the bait to play against
            if snake_opponent_pool.should_use_historical():
                using_historical_snake = True
                original_snake_policy = {k: v.clone() for k, v in
                                         snake_agent.policy_net.state_dict().items()}
                snake_agent.policy_net.load_state_dict(
                    snake_opponent_pool.get_random_snapshot()
                )

        for step in range(config.game.max_steps):
            snake_action = snake_agent.select_action(snake_obs)
            bait_action  = bait_agent.select_action(bait_obs)

            (snake_obs_next, bait_obs_next,
             snake_reward, bait_reward, done, info) = env.step(snake_action, bait_action)

            if is_multi_discrete:
                s_tool = info.get("snake_tool", "NONE")
                b_tool = info.get("bait_tool",  "NONE")
                snake_tool_counts[s_tool] = snake_tool_counts.get(s_tool, 0) + 1
                bait_tool_counts[b_tool]  = bait_tool_counts.get(b_tool, 0) + 1

            # ── Reward clipping ────────────────────────────────────────────
            snake_reward = max(-coevo.reward_clip,
                               min(coevo.reward_clip, snake_reward))
            bait_reward  = max(-coevo.reward_clip,
                               min(coevo.reward_clip, bait_reward))

            # ── Reward normalisation ───────────────────────────────────────
            if coevo.reward_normalize:
                norm_snake_r = snake_reward_norm.normalize(snake_reward)
                norm_bait_r  = bait_reward_norm.normalize(bait_reward)
            else:
                norm_snake_r = snake_reward
                norm_bait_r  = bait_reward

            # Store normalised rewards in replay buffer (only for the
            # agent whose policy is LIVE, not the historical snapshot)
            if not using_historical_snake:
                snake_agent.store_transition(
                    snake_obs, snake_action, norm_snake_r, snake_obs_next, done)
            if not using_historical_bait:
                bait_agent.store_transition(
                    bait_obs, bait_action, norm_bait_r, bait_obs_next, done)

            # ── Training steps ─────────────────────────────────────────────
            if not using_historical_snake:
                s_loss = snake_agent.train_step()
            else:
                s_loss = None

            if not using_historical_bait:
                b_loss = bait_agent.train_step()
            else:
                b_loss = None

            if s_loss is not None or b_loss is not None:
                ep_snake_loss += s_loss if s_loss else 0
                ep_bait_loss  += b_loss if b_loss else 0
                loss_count    += 1

            ep_snake_reward += snake_reward   # Track raw reward for logging
            ep_bait_reward  += bait_reward

            snake_obs = snake_obs_next
            bait_obs  = bait_obs_next

            if done:
                break

        # ── Restore original policies if using historical opponents ────────
        if using_historical_bait and original_bait_policy is not None:
            bait_agent.policy_net.load_state_dict(original_bait_policy)
        if using_historical_snake and original_snake_policy is not None:
            snake_agent.policy_net.load_state_dict(original_snake_policy)

        # ── Opponent pool: periodically save snapshots ─────────────────────
        if (coevo.use_opponent_pool and
                snake_agent.steps_done % coevo.pool_save_interval == 0 and
                snake_agent.steps_done > 0):
            snake_opponent_pool.save_snapshot(snake_agent)
            bait_opponent_pool.save_snapshot(bait_agent)

        # ── Post-Episode ──────────────────────────────────────────────────────
        total_games += 1
        if info.get("event") == "capture":
            snake_wins += 1

        avg_s_loss = ep_snake_loss / max(loss_count, 1)
        avg_b_loss = ep_bait_loss  / max(loss_count, 1)

        # NOTE: Hard target sync is REMOVED. Soft update (Polyak τ) happens
        # inside dqn_agent.train_step() on every gradient step. The old
        # sync_target_network() method is kept for backward compatibility
        # but is no longer called by default.

        tool_counts = (
            {"snake": snake_tool_counts, "bait": bait_tool_counts}
            if is_multi_discrete else None
        )

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

        # ── W&B per-episode logging ───────────────────────────────────────────
        win_rate = snake_wins / max(total_games, 1) * 100
        wandb_metrics = {
            "episode":              episode,
            "snake/reward":         ep_snake_reward,
            "bait/reward":          ep_bait_reward,
            "snake/loss":           avg_s_loss,
            "bait/loss":            avg_b_loss,
            "snake/epsilon":        snake_agent.epsilon,
            "snake/steps_done":     snake_agent.steps_done,
            "snake/lr":             snake_agent.optimizer.param_groups[0]["lr"],
            "episode/steps":        step + 1,
            "episode/win_rate_pct": win_rate,
            "episode/winner":       1 if info.get("event") == "capture" else 0,
            "coevo/using_hist_snake": int(using_historical_snake),
            "coevo/using_hist_bait":  int(using_historical_bait),
            "coevo/snake_pool_size":  len(snake_opponent_pool),
            "coevo/bait_pool_size":   len(bait_opponent_pool),
        }
        # Log tool usage in multi-discrete mode
        if is_multi_discrete:
            for tool_name, count in snake_tool_counts.items():
                wandb_metrics[f"snake/tool_{tool_name}"] = count
            for tool_name, count in bait_tool_counts.items():
                wandb_metrics[f"bait/tool_{tool_name}"] = count

        # ── Live Visual Telemetry (Every 1000 Episodes) ───────────────────────
        if episode % 1000 == 0 and _WANDB_AVAILABLE:
            try:
                # Get frames (H, W, 3) arrays
                frames = env.record_eval_episode(snake_agent, bait_agent)
                # wandb.Video expects (time, channel, height, width)
                # so we transpose (H, W, C) -> (C, H, W) for each frame
                vid_frames = np.array([np.transpose(f, (2, 0, 1)) for f in frames])
                wandb_metrics["live_observation"] = wandb.Video(vid_frames, fps=10, format="gif")
            except Exception as e:
                log.warning(f"Failed to record live telemetry video: {e}")

        _wandb_log(wandb_metrics)

        # ── Console progress ──────────────────────────────────────────────────
        if episode % config.training.log_interval == 0:
            now            = time.time()
            elapsed        = now - _last_log_time
            _last_log_time = now
            mins, secs     = divmod(int(elapsed), 60)
            time_str       = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

            lr_str = f"{snake_agent.optimizer.param_groups[0]['lr']:.2e}"
            hist_marker = ""
            if using_historical_snake:
                hist_marker += " 📦🐍"
            if using_historical_bait:
                hist_marker += " 📦🎯"
            print(f"  ⏱  +{time_str}")
            print(
                f"Ep {episode:>7,d} | "
                f"ε={snake_agent.epsilon:.3f} | "
                f"🐍 R={ep_snake_reward:>7.2f} | "
                f"🎯 R={ep_bait_reward:>7.2f} | "
                f"Win%={win_rate:>5.1f}% | "
                f"Steps={step+1:>3d} | "
                f"Loss={avg_s_loss:.4f} | "
                f"LR={lr_str}"
                f"{hist_marker}"
            )
            if is_multi_discrete:
                s_top = max(snake_tool_counts, key=snake_tool_counts.get)
                b_top = max(bait_tool_counts,  key=bait_tool_counts.get)
                print(
                    f"         🛠 Snake top-tool: {s_top}={snake_tool_counts[s_top]} | "
                    f"Bait top-tool: {b_top}={bait_tool_counts[b_top]}"
                )
            snake_wins  = 0
            total_games = 0

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if episode % ckpt_cfg.interval == 0:
            save_path = manager.save(
                episode       = episode,
                snake_agent   = snake_agent,
                bait_agent    = bait_agent,
                training_meta = _build_training_meta(),
            )
            print(f"  💾 Checkpoint saved → {save_path} (episode {episode:,d})")
            # Log checkpoint milestone to W&B
            _wandb_log({"checkpoint/episode": episode, "checkpoint/path": str(save_path)})

    # ─────────────────────────────────────────────────────────────────────────
    #  Final save
    # ─────────────────────────────────────────────────────────────────────────
    final_path = manager.save(
        episode       = config.training.num_episodes,
        snake_agent   = snake_agent,
        bait_agent    = bait_agent,
        training_meta = {
            **_build_training_meta(),
            "final": True,
        },
    )
    snake_agent.save(f"{ckpt_cfg.checkpoint_dir}/snake_final.pt")
    bait_agent.save(f"{ckpt_cfg.checkpoint_dir}/bait_final.pt")

    logger.close()

    _wandb_log({"training/complete": True, "training/final_episode": config.training.num_episodes})

    print(f"\n✅ Training complete!  Final checkpoint → {final_path}")
    print(f"   Legacy flat files  → {ckpt_cfg.checkpoint_dir}/snake_final.pt")