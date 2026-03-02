"""
main.py — Command-line entry point for NeuroEvasion.

Usage:
    python main.py train                    # Train agents
    python main.py train --episodes 10000   # Custom episode count
    python main.py demo                     # Watch trained agents play
    python main.py evaluate                 # Evaluate model performance
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
