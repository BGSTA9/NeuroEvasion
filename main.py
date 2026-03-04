"""
main.py — Command-line entry point for NeuroEvasion.

Usage:
    python main.py train                             # Train agents (auto-resume)
    python main.py train --no-resume                 # Force a fresh start
    python main.py train --episodes 10000            # Custom episode count
    python main.py train --checkpoint-interval 500   # Save every 500 episodes
    python main.py train --drive-sync-dir /content/drive/MyDrive/NeuroEvasion/ckpts
    python main.py demo --snake-model ...            # Watch trained agents play
    python main.py evaluate --snake-model ...        # Evaluate model performance
"""

import argparse
import sys
from config import Config


def main():
    parser = argparse.ArgumentParser(
        description="🧠 NeuroEvasion — Co-Evolutionary Pursuit-Evasion with DNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Train command ---
    train_parser = subparsers.add_parser("train", help="Train the agents")
    train_parser.add_argument("--episodes", type=int, default=None,
                              help="Number of training episodes")
    train_parser.add_argument("--grid-size", type=int, default=None,
                              help="Grid size (N×N)")
    train_parser.add_argument("--lr", type=float, default=None,
                              help="Learning rate")
    train_parser.add_argument("--device", type=str, default=None,
                              choices=["cpu", "cuda"],
                              help="Training device")
    train_parser.add_argument("--seed", type=int, default=None,
                              help="Random seed")

    # Checkpoint / resume control
    resume_group = train_parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true",
                              default=True,
                              help="Auto-resume from latest checkpoint (default)")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false",
                              help="Ignore existing checkpoints and start fresh")
    train_parser.add_argument("--checkpoint-dir", type=str, default=None,
                              metavar="PATH",
                              help="Directory to save/load checkpoints (default: checkpoints/)")
    train_parser.add_argument("--checkpoint-interval", type=int, default=None,
                              metavar="N",
                              help="Save a checkpoint every N episodes")
    train_parser.add_argument("--keep-last-n", type=int, default=None,
                              metavar="N",
                              help="Number of recent checkpoints to keep on disk")
    train_parser.add_argument("--drive-sync-dir", type=str, default=None,
                              metavar="PATH",
                              help="Mirror checkpoints here after each save (e.g. Google Drive)")

    # Multi-Discrete action space
    train_parser.add_argument("--multi-discrete", dest="multi_discrete",
                              action="store_true", default=False,
                              help="Enable dual-head Multi-Discrete agents "
                                   "(move + tool-use per step)")
    train_parser.add_argument("--dueling", dest="dueling",
                              action="store_true", default=False,
                              help="Use Dueling DQN decomposition in each action head")

    # --- Demo command ---
    demo_parser = subparsers.add_parser("demo", help="Watch agents play")
    demo_parser.add_argument("--snake-model", type=str, required=True,
                             help="Path to snake checkpoint")
    demo_parser.add_argument("--bait-model", type=str, required=True,
                             help="Path to bait checkpoint")
    demo_parser.add_argument("--speed", type=int, default=10,
                             help="Game speed (FPS)")

    # --- Evaluate command ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate agents")
    eval_parser.add_argument("--snake-model", type=str, required=True)
    eval_parser.add_argument("--bait-model", type=str, required=True)
    eval_parser.add_argument("--num-games", type=int, default=1000)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    config = Config()

    if args.command == "train":
        if args.episodes:
            config.training.num_episodes = args.episodes
        if args.grid_size:
            config.game.grid_size = args.grid_size
        if args.lr:
            config.agent.learning_rate = args.lr
        if args.device:
            config.training.device = args.device
        if args.seed:
            config.seed = args.seed

        # Apply checkpoint overrides
        if args.checkpoint_dir:
            config.checkpoint.checkpoint_dir = args.checkpoint_dir
        if args.checkpoint_interval:
            config.checkpoint.interval = args.checkpoint_interval
        if args.keep_last_n is not None:
            config.checkpoint.keep_last_n = args.keep_last_n
        if args.drive_sync_dir:
            config.checkpoint.drive_sync_dir = args.drive_sync_dir

        # --no-resume: wipe checkpoint dir from manager's perspective by
        # pointing it at a fresh subdirectory so load_latest() returns None
        if not args.resume:
            import time as _time
            config.checkpoint.checkpoint_dir = (
                f"{config.checkpoint.checkpoint_dir}/run_{int(_time.time())}"
            )
            print("🆕 --no-resume: starting a fresh training run.")

        # Multi-Discrete flags
        if args.multi_discrete:
            config.multi_discrete.use_multi_discrete = True
        if args.dueling:
            config.multi_discrete.use_dueling = True

        from training.trainer import train
        train(config)

    elif args.command == "demo":
        from visualization.renderer import run_demo
        run_demo(config, args.snake_model, args.bait_model, args.speed)

    elif args.command == "evaluate":
        from evaluation.evaluator import evaluate
        evaluate(config, args.snake_model, args.bait_model, args.num_games)


if __name__ == "__main__":
    main()
