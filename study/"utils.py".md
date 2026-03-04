# 📄 utils.py — What Does Every Line Do?

> 🎯 **Goal of this file:** Make sure that every time we run our program, it behaves **exactly the same way**. No surprises!

---

## 🗒️ The Note at the Top

```python
"""
utils.py — Utilities for reproducibility and common operations.
"""
```

This is just a **sticky note** on the file. It tells anyone who opens it:
*"Hey! This file has helpful tools, especially for making sure the program always runs the same way."*

The computer doesn't do anything with it — it's only for humans to read.

---

## 📦 Bringing in Tools

```python
import random
```
This brings in a **tool that can pick random numbers**, like rolling a dice. Python has this built in, but we need to tell it to wake up first.

---

```python
import numpy as np
```
This brings in **NumPy** — a super powerful math tool. It has its *own* dice inside it, separate from Python's dice. We nickname it `np` so we don't have to type the full name every time.

---

```python
import torch
```
This brings in **PyTorch** — the tool that builds and trains our neural network (the "brain" of our AI). It also has its *own* dice. So now we have **3 separate dice** in total.

---

## 🎲 Why Do We Have 3 Separate Dice?

Imagine you have 3 friends, and each one has their own dice:
- 🎲 **Python's dice** (`random`)
- 🎲 **NumPy's dice** (`np.random`)
- 🎲 **PyTorch's dice** (`torch`)

If you want everyone to get the **same result every time**, you have to tell **each friend** what number to start from. You can't just tell one of them — you have to tell all three!

---

## 🔧 The Function

```python
def set_global_seed(seed: int) -> None:
```

This creates a **recipe** called `set_global_seed`. A recipe is a set of steps the computer follows when you call its name.

- `seed` is an **ingredient** — it's a number you give it (like `42`).
- `: int` means the ingredient must be a **whole number** (1, 2, 42, 100... not 3.5).
- `-> None` means this recipe **doesn't give anything back** — it just does its job quietly.

---

```python
    """
    Set seeds for all random number generators.
    ...
    """
```

Another **sticky note**, but this one is inside the recipe. It explains what the recipe does and *why* it matters. Still just for humans — the computer skips it.

---

## 🪄 Inside the Recipe — The Steps

```python
    random.seed(seed)
```
We walk up to **Python's dice** and say: *"Start counting from this number."*
Now whenever Python rolls its dice, we know exactly what numbers will come out.

---

```python
    np.random.seed(seed)
```
We walk up to **NumPy's dice** and say the same thing: *"Start counting from this number."*
NumPy is a separate friend, so we have to tell it separately!

---

```python
    torch.manual_seed(seed)
```
We walk up to **PyTorch's dice** (the CPU one) and say: *"Start counting from this number."*
Now the AI brain's randomness on the regular computer chip is locked in too.

---

```python
    if torch.cuda.is_available():
```
This is a **question**: *"Hey, is there a super-fast graphics card (GPU) plugged in?"*

- If **yes** → keep going with the steps below.
- If **no** → skip all the steps below.

A GPU is like a turbo engine that makes AI training go much faster. But it has its *own* dice too!

---

```python
        torch.cuda.manual_seed_all(seed)
```
We walk up to **the GPU's dice** and say: *"Start counting from this number too."*

`all` means even if there are **multiple GPUs**, we set the same starting point for all of them at once.

---

```python
        # These ensure deterministic behavior on GPU (may reduce performance)
```
This is a **comment** — a note left by the programmer. The computer ignores it completely. It's warning us: *"The next two lines make things predictable, but might make the program a tiny bit slower."*

---

```python
        torch.backends.cudnn.deterministic = True
```
Deep inside PyTorch, there's a helper called **cuDNN** that runs things on the GPU. Normally, cuDNN likes to take shortcuts to go faster — but those shortcuts can give slightly different results each time.

Setting this to `True` tells cuDNN: **"No shortcuts. Do it the same way, every time."**

Think of it like telling someone: *"Don't improvise — follow the recipe exactly."*

---

```python
        torch.backends.cudnn.benchmark = False
```
cuDNN also likes to **experiment** before starting — it tries a few different methods and picks the fastest one. But that experiment can give different results on different computers.

Setting this to `False` tells cuDNN: **"Don't experiment. Just use the standard method."**

---

## 🧠 The Big Picture — Why Does All This Matter?

Imagine you're baking a cake and you write down the recipe. Next week, your friend tries to bake the **exact same cake** using your recipe — but it comes out different! Why? Because you both started with different random choices (how much you stirred, which oven rack you used, etc.).

In AI, the same problem happens. The program makes **hundreds of random choices** while training:
- How to set up the brain at the start
- Which examples to practice with first
- Which random actions to try

If we don't fix all the dice to start from the same number, **two runs of the same code can give different results** — which makes it really hard to learn from or compare experiments.

`set_global_seed(42)` is like saying: **"Everyone, start from 42. No exceptions."**
Now the cake comes out the same way, every single time. 🎂

---

## 📋 Quick Cheat Sheet

| Line | What it does in simple words |
|------|------------------------------|
| `import random` | Wake up Python's dice |
| `import numpy as np` | Wake up NumPy's dice, call it `np` |
| `import torch` | Wake up PyTorch (the AI brain tool) |
| `def set_global_seed(seed):` | Create a recipe that needs a number |
| `random.seed(seed)` | Tell Python's dice: start from this number |
| `np.random.seed(seed)` | Tell NumPy's dice: start from this number |
| `torch.manual_seed(seed)` | Tell PyTorch's CPU dice: start from this number |
| `if torch.cuda.is_available():` | Ask: is there a GPU plugged in? |
| `torch.cuda.manual_seed_all(seed)` | Tell ALL GPU dice: start from this number |
| `cudnn.deterministic = True` | Tell the GPU helper: no shortcuts, be consistent |
| `cudnn.benchmark = False` | Tell the GPU helper: don't experiment, use standard method |