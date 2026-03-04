# `utils.py` — Line-by-Line Breakdown

---

## Module Docstring

```python
"""
utils.py — Utilities for reproducibility and common operations.
"""
```
| Line | Description |
|------|-------------|
| `"""..."""` | A **module-level docstring**. Purely informational — describes the file's purpose. Python stores it in `__doc__` but it has no runtime effect. |

---

## Imports

```python
import random
import numpy as np
import torch
```

| Line | Description |
|------|-------------|
| `import random` | Imports Python's **built-in random module**, which controls Python-native random number generation (e.g. `random.choice`, `random.random`). |
| `import numpy as np` | Imports **NumPy** and aliases it as `np`. NumPy has its own internal random number generator (RNG), separate from Python's. |
| `import torch` | Imports **PyTorch**, the deep learning framework. PyTorch also maintains its own RNG — both for CPU and GPU operations. |

---

## Function Signature & Docstring

```python
def set_global_seed(seed: int) -> None:
```

| Line | Description |
|------|-------------|
| `def set_global_seed(...)` | Declares a function named `set_global_seed`. |
| `seed: int` | **Type hint** — declares that the `seed` parameter is expected to be an integer. Not enforced at runtime, but aids readability and tooling. |
| `-> None` | **Return type hint** — declares that this function returns nothing. |

```python
    """
    Set seeds for all random number generators.

    WHY THIS MATTERS:
        Neural network training involves randomness at many levels:
        - Weight initialization
        - Replay buffer sampling
        - Epsilon-greedy action selection
        - Environment resets

        Fixing all seeds makes experiments reproducible:
        same seed → same results, every time.
    """
```

| Line | Description |
|------|-------------|
| `"""..."""` | A **function-level docstring**. Describes what the function does and explains *why* seeding matters in neural network training. Accessible via `set_global_seed.__doc__`. |

---

## Function Body

```python
    random.seed(seed)
```
| Line | Description |
|------|-------------|
| `random.seed(seed)` | Sets the seed for Python's **built-in `random` module**. Any subsequent calls to `random.*` (e.g. `random.randint`, `random.shuffle`) will produce a deterministic sequence starting from this seed. |

---

```python
    np.random.seed(seed)
```
| Line | Description |
|------|-------------|
| `np.random.seed(seed)` | Sets the seed for **NumPy's global RNG**. Affects all NumPy random operations such as `np.random.choice`, `np.random.randn`, etc. NumPy's RNG is independent of Python's, so it must be seeded separately. |

---

```python
    torch.manual_seed(seed)
```
| Line | Description |
|------|-------------|
| `torch.manual_seed(seed)` | Sets the seed for **PyTorch's CPU RNG**. Controls randomness in CPU-based tensor operations — including weight initialization, dropout masks, and data shuffling done through PyTorch. |

---

```python
    if torch.cuda.is_available():
```
| Line | Description |
|------|-------------|
| `torch.cuda.is_available()` | Checks whether a **CUDA-capable GPU** is present and accessible. Returns `True` if a GPU is available, `False` otherwise. The block below only runs when a GPU exists. |

---

```python
        torch.cuda.manual_seed_all(seed)
```
| Line | Description |
|------|-------------|
| `torch.cuda.manual_seed_all(seed)` | Sets the seed for the **PyTorch GPU RNG across all available GPUs**. Without this, GPU operations (e.g. CUDA kernels for conv layers, matrix multiplications) would still be non-deterministic even if the CPU seed is fixed. |

---

```python
        # These ensure deterministic behavior on GPU (may reduce performance)
```
| Line | Description |
|------|-------------|
| `# These ensure...` | An **inline comment** warning that the next two lines trade speed for determinism. No executable effect. |

---

```python
        torch.backends.cudnn.deterministic = True
```
| Line | Description |
|------|-------------|
| `torch.backends.cudnn.deterministic = True` | Instructs **cuDNN** (NVIDIA's GPU deep learning library used internally by PyTorch) to only use deterministic algorithms. By default, cuDNN may pick faster non-deterministic algorithms. Setting this to `True` disables that, ensuring identical results across runs at the cost of some speed. |

---

```python
        torch.backends.cudnn.benchmark = False
```
| Line | Description |
|------|-------------|
| `torch.backends.cudnn.benchmark = False` | Disables cuDNN's **auto-tuner**, which normally benchmarks multiple algorithm implementations at startup and picks the fastest one. That selection process is non-deterministic and input-size-dependent. Setting this to `False` prevents it, which is required for full reproducibility (and pairs with `deterministic = True` above). |