"""Reproducible seeding across random / numpy / torch (+ CUDA)."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int = 1337, *, deterministic: bool = False) -> int:
    """Seed Python, numpy and (if installed) torch. Returns the seed used.

    With ``deterministic=True`` the strictest cuDNN/torch determinism flags are
    set. That trades throughput for bit-reproducibility and is intended for
    regression tests, not production training runs.
    """
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    if type(deterministic) is not bool:
        raise TypeError("deterministic must be a boolean")
    # This controls hash randomization in child interpreters. Python's hash seed
    # for the current process is fixed before startup and cannot be changed here.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a core dep
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
    except ImportError:
        pass

    return seed


__all__ = ["seed_everything"]
