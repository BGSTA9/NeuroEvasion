# 📄 main.py — What Does Every Line Do?

> 🎯 **Goal of this file:** This is the **front door** of the whole program. When you want to train the AI, watch it play, or test how good it is — you come here first and tell it what to do.

---

## 🗒️ The Note at the Top

```python
"""
main.py — Command-line entry point for NeuroEvasion.

Usage:
    python main.py train
    python main.py train --no-resume
    ...
"""
```

This is a big **sticky note** at the top of the file. It shows examples of how to use the program by typing commands into the terminal (the black screen where you type instructions to the computer).

Nobody has to read it — but it's super helpful for humans who are new to the project.

---

## 📦 Bringing in Tools

```python
import argparse
```
`argparse` is a built-in Python tool that helps the program **read and understand commands you type**. Think of it like a receptionist — when you walk in and say *"I want to train!"*, `argparse` hears it, understands it, and passes the message along correctly.

---

```python
import sys
```
`sys` is a built-in tool that lets the program **talk to the operating system**. Here we use it for one specific job: **closing the program** when something goes wrong or when no command was given.

---

```python
from config import Config
```
This reaches into a file called `config.py` and pulls out a blueprint called `Config`. Think of `Config` as a **settings book** — it holds all the default numbers and options for the whole project (like grid size, learning rate, etc.).

---

## 🔧 The Main Recipe

```python
def main():
```
This creates the **main recipe** — the set of steps that runs when you launch the program. Everything else lives inside here.

---

## 🗣️ Setting Up the Receptionist

```python
    parser = argparse.ArgumentParser(
        description="🧠 NeuroEvasion — Co-Evolutionary Pursuit-Evasion with DNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
```
This creates the **receptionist** (`parser`). 

- `description=` is the welcome sign on the receptionist's desk — it tells you what the program is about when you ask for help.
- `formatter_class=argparse.RawDescriptionHelpFormatter` tells the receptionist to display the help text *exactly as written* (keeping spaces and line breaks tidy).

---

```python
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
```
This tells the receptionist: *"Get ready — people will walk in and say one of a few different things."*

- `dest="command"` means whatever the person types first (like `train`, `demo`, or `evaluate`) gets saved under the name `"command"` so we can check it later.
- Think of `subparsers` as a **menu board** — each item on the menu is a different thing you can do.

---

## 🏋️ The "Train" Menu Item

```python
    train_parser = subparsers.add_parser("train", help="Train the agents")
```
Adds **"train"** to the menu. If someone types `python main.py train`, this item handles it. `train_parser` is now its own mini-receptionist just for training questions.

---

```python
    train_parser.add_argument("--episodes", type=int, default=None,
                              help="Number of training episodes")
```
Adds an **optional extra instruction** called `--episodes`. 
- `type=int` → must be a whole number.
- `default=None` → if you don't mention it, the program pretends you didn't say anything (and uses the default from the settings book instead).
- Think of it like saying: *"How many rounds of practice do you want?"*

---

```python
    train_parser.add_argument("--grid-size", type=int, default=None,
                              help="Grid size (N×N)")
```
Optional instruction: **how big should the game board be?** (e.g. `--grid-size 10` means a 10×10 board).

---

```python
    train_parser.add_argument("--lr", type=float, default=None,
                              help="Learning rate")
```
Optional instruction: **how fast should the AI learn?** (`float` means it can be a decimal like `0.001`). The learning rate controls how big the steps are when the AI updates its brain.

---

```python
    train_parser.add_argument("--device", type=str, default=None,
                              choices=["cpu", "cuda"],
                              help="Training device")
```
Optional instruction: **which chip should do the math?**
- `"cpu"` = the regular computer brain.
- `"cuda"` = the turbo graphics card (GPU).
- `choices=` means only these two answers are allowed — anything else causes an error.

---

```python
    train_parser.add_argument("--seed", type=int, default=None,
                              help="Random seed")
```
Optional instruction: **give the dice a starting number** so the training is reproducible (same seed = same results every time — remember `utils.py`? 🎲).

---

## 💾 Checkpoint (Save File) Controls

```python
    resume_group = train_parser.add_mutually_exclusive_group()
```
Creates a **special group** where only ONE of the options inside it can be chosen at a time. Like a light switch — it's either ON or OFF, never both.

---

```python
    resume_group.add_argument("--resume", dest="resume", action="store_true",
                              default=True,
                              help="Auto-resume from latest checkpoint (default)")
```
If you type `--resume` (or say nothing at all, since `default=True`), the program will **pick up where it last left off** — like loading a saved game.

- `action="store_true"` → typing this flag sets `resume = True`.
- `dest="resume"` → saves the answer under the name `"resume"`.

---

```python
    resume_group.add_argument("--no-resume", dest="resume", action="store_false",
                              help="Ignore existing checkpoints and start fresh")
```
If you type `--no-resume`, the program will **ignore all saved progress and start brand new** — like deleting your save file and playing from the beginning.

- `action="store_false"` → typing this flag sets `resume = False`.
- Both `--resume` and `--no-resume` write to the same `dest="resume"` — that's the switch!

---

```python
    train_parser.add_argument("--checkpoint-dir", type=str, default=None,
                              metavar="PATH",
                              help="Directory to save/load checkpoints (default: checkpoints/)")
```
Optional instruction: **which folder should save files go into?** A "checkpoint" is like a save file for the AI's brain. `metavar="PATH"` just changes how the help text looks (shows `PATH` instead of `CHECKPOINT_DIR`).

---

```python
    train_parser.add_argument("--checkpoint-interval", type=int, default=None,
                              metavar="N",
                              help="Save a checkpoint every N episodes")
```
Optional instruction: **how often should we create a save file?** e.g. `--checkpoint-interval 500` = save every 500 episodes of practice.

---

```python
    train_parser.add_argument("--keep-last-n", type=int, default=None,
                              metavar="N",
                              help="Number of recent checkpoints to keep on disk")
```
Optional instruction: **how many old save files should we keep?** Old ones get deleted to save storage space. Like only keeping your last 3 attempts at a level.

---

```python
    train_parser.add_argument("--drive-sync-dir", type=str, default=None,
                              metavar="PATH",
                              help="Mirror checkpoints here after each save (e.g. Google Drive)")
```
Optional instruction: **copy every save file to Google Drive too**, so you don't lose progress if the computer crashes. It's like having a backup copy of your save file in the cloud ☁️.

---

## 🕹️ Multi-Discrete & Dueling Flags

```python
    train_parser.add_argument("--multi-discrete", dest="multi_discrete",
                              action="store_true", default=False,
                              help="Enable dual-head Multi-Discrete agents "
                                   "(move + tool-use per step)")
```
Optional flag: **turn on a fancier version of the AI** where each agent makes *two decisions at once* per step (where to move AND which tool to use). Off by default.

---

```python
    train_parser.add_argument("--dueling", dest="dueling",
                              action="store_true", default=False,
                              help="Use Dueling DQN decomposition in each action head")
```
Optional flag: **use a smarter brain design** called Dueling DQN — it splits the brain into two parts to help the AI learn faster and more accurately. Off by default.

---

## 🎮 The "Demo" Menu Item

```python
    demo_parser = subparsers.add_parser("demo", help="Watch agents play")
```
Adds **"demo"** to the menu. This mode lets you **watch the trained AI play the game** visually, like watching a replay.

---

```python
    demo_parser.add_argument("--snake-model", type=str, required=True,
                             help="Path to snake checkpoint")
```
**Required** instruction: tell the program *where the snake AI's save file is*. `required=True` means you MUST provide this — the program will yell at you if you forget.

---

```python
    demo_parser.add_argument("--bait-model", type=str, required=True,
                             help="Path to bait checkpoint")
```
**Required** instruction: tell the program *where the bait AI's save file is*. Also required — both players need to be loaded to play!

---

```python
    demo_parser.add_argument("--speed", type=int, default=10,
                             help="Game speed (FPS)")
```
Optional instruction: **how fast should the game play?** Measured in frames per second (FPS). Default is 10 — like choosing slow-mo or fast-forward when watching a video.

---

## 📊 The "Evaluate" Menu Item

```python
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate agents")
```
Adds **"evaluate"** to the menu. This mode runs the AI through many games and **measures how good it is** — no visuals, just numbers and stats.

---

```python
    eval_parser.add_argument("--snake-model", type=str, required=True)
    eval_parser.add_argument("--bait-model", type=str, required=True)
```
Both model paths are **required** — same idea as demo mode, you need to load both AIs.

---

```python
    eval_parser.add_argument("--num-games", type=int, default=1000)
```
Optional instruction: **how many games should we test over?** Default is 1000 games. The more games, the more reliable the score.

---

## 🚦 Reading What the User Actually Typed

```python
    args = parser.parse_args()
```
This is the moment the receptionist **actually reads what you typed**. All those instructions we defined above? Now `argparse` scans your command and fills them all in. The result is stored in `args` — a little bag holding all your answers.

---

```python
    if args.command is None:
        parser.print_help()
        sys.exit(1)
```
If you typed `python main.py` with **nothing after it**, the receptionist doesn't know what you want.
- `parser.print_help()` → prints the menu so you know your options.
- `sys.exit(1)` → closes the program. The `1` signals that something went wrong (as opposed to `0` which means "everything is fine").

---

## ⚙️ Setting Up the Settings Book

```python
    config = Config()
```
Opens the **settings book** with all the default values. Now we'll update it based on whatever the user typed.

---

## 🏋️ What Happens When You Choose "train"

```python
    if args.command == "train":
```
Checks: *did the user type `train`?* If yes, run everything inside this block.

---

```python
        if args.episodes:
            config.training.num_episodes = args.episodes
```
If the user gave a number of episodes → **write it into the settings book**. Otherwise, leave the default.

---

```python
        if args.grid_size:
            config.game.grid_size = args.grid_size
```
If the user gave a grid size → **update the settings book** with it.

---

```python
        if args.lr:
            config.agent.learning_rate = args.lr
```
If the user gave a learning rate → **update the settings book** with it.

---

```python
        if args.device:
            config.training.device = args.device
```
If the user chose `cpu` or `cuda` → **update the settings book** with it.

---

```python
        if args.seed:
            config.seed = args.seed
```
If the user gave a seed number → **update the settings book** with it.

---

```python
        if args.checkpoint_dir:
            config.checkpoint.checkpoint_dir = args.checkpoint_dir
        if args.checkpoint_interval:
            config.checkpoint.interval = args.checkpoint_interval
        if args.keep_last_n is not None:
            config.checkpoint.keep_last_n = args.keep_last_n
        if args.drive_sync_dir:
            config.checkpoint.drive_sync_dir = args.drive_sync_dir
```
Each of these checks if the user gave a checkpoint-related instruction, and if so, **writes it into the settings book**. Note: `keep_last_n` checks `is not None` specifically because `0` is a valid value (keep zero old saves), and `if 0:` would wrongly be treated as "nothing given."

---

```python
        if not args.resume:
            import time as _time
            config.checkpoint.checkpoint_dir = (
                f"{config.checkpoint.checkpoint_dir}/run_{int(_time.time())}"
            )
            print("🆕 --no-resume: starting a fresh training run.")
```
If the user typed `--no-resume`:
- `import time as _time` → grabs the clock tool (imported here instead of the top because it's only needed in this rare case).
- Creates a **brand new folder** with the current timestamp in its name (like `checkpoints/run_1712345678`). Since no old checkpoints are in this new folder, the AI can't find any saves and has to start fresh.
- Prints a message so the user knows it worked. 🆕

---

```python
        if args.multi_discrete:
            config.multi_discrete.use_multi_discrete = True
        if args.dueling:
            config.multi_discrete.use_dueling = True
```
If the user turned on multi-discrete or dueling flags → **flip those switches in the settings book**.

---

```python
        from training.trainer import train
        train(config)
```
- Reaches into the `training/trainer.py` file and pulls out the `train` function (imported here, not at the top, so we only load it when actually needed).
- **Starts training!** Hands the settings book (`config`) to the trainer so it knows all the rules.

---

## 🎮 What Happens When You Choose "demo"

```python
    elif args.command == "demo":
        from visualization.renderer import run_demo
        run_demo(config, args.snake_model, args.bait_model, args.speed)
```
- Loads the `run_demo` function from the renderer file.
- **Launches the visual demo** — passes the settings book, both model paths, and the desired speed.

---

## 📊 What Happens When You Choose "evaluate"

```python
    elif args.command == "evaluate":
        from evaluation.evaluator import evaluate
        evaluate(config, args.snake_model, args.bait_model, args.num_games)
```
- Loads the `evaluate` function from the evaluator file.
- **Runs the evaluation** — passes the settings book, both model paths, and how many games to test over.

---

## 🚪 The Real Front Door

```python
if __name__ == "__main__":
    main()
```
This is Python's special way of saying: **"Only run `main()` if this file was launched directly."**

- If someone types `python main.py train` in the terminal → `__name__` equals `"__main__"` → `main()` runs. ✅
- If some other file imports `main.py` for its tools → `__name__` is NOT `"__main__"` → `main()` does NOT run automatically. ✅

Think of it like a light switch on the front door: the lights only turn on when *you* open the door, not when someone just peeks through the window.

---

## 🧠 The Big Picture — How It All Fits Together

```
You type a command in the terminal
        ↓
argparse reads it (the receptionist)
        ↓
Config() opens the settings book with defaults
        ↓
Your custom options overwrite the defaults
        ↓
   ┌────────────────────────────────────┐
   │  train  →  trainer.train(config)  │
   │  demo   →  renderer.run_demo(...) │
   │  evaluate → evaluator.evaluate(…) │
   └────────────────────────────────────┘
```

`main.py` doesn't do any of the hard work itself — it just **reads your wishes, prepares the settings, and calls the right helper** to do the actual job.

---

## 📋 Quick Cheat Sheet

| Line | What it does in simple words |
|------|------------------------------|
| `import argparse` | Wake up the "receptionist" tool |
| `import sys` | Wake up the tool that can close the program |
| `from config import Config` | Grab the settings book blueprint |
| `def main():` | Create the main recipe |
| `ArgumentParser(...)` | Build the receptionist's desk |
| `add_subparsers(...)` | Create the menu board |
| `add_parser("train")` | Add "train" to the menu |
| `add_argument("--episodes", ...)` | Add an optional "how many episodes?" slot |
| `add_mutually_exclusive_group()` | Create an either/or switch |
| `--resume / --no-resume` | Load save file OR start fresh |
| `--drive-sync-dir` | Backup saves to Google Drive |
| `--multi-discrete` | Turn on fancy dual-decision AI |
| `--dueling` | Turn on smarter brain design |
| `parse_args()` | Receptionist reads what you actually typed |
| `sys.exit(1)` | Close the program (something went wrong) |
| `config = Config()` | Open the settings book |
| `config.X = args.X` | Write user's choices into the settings book |
| `run_{int(_time.time())}` | Make a fresh folder using the current time |
| `from training.trainer import train` | Load the trainer tool only when needed |
| `train(config)` | Start training the AI! |
| `if __name__ == "__main__":` | Only run if this file is the front door |