from __future__ import annotations

import importlib
import importlib.util
import random
from typing import Any, Optional

import numpy as np

from .config import PipelineConfig


def _load_torch() -> Optional[Any]:
    """Load torch if available without forcing a hard dependency."""
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return None
    return importlib.import_module("torch")


def apply_determinism(config: PipelineConfig) -> None:
    """
    Apply deterministic settings based on PipelineConfig.

    When a seed or deterministic run is requested, this seeds Python, NumPy,
    and (optionally) torch. Torch deterministic algorithms are enabled only
    when explicitly requested via config.deterministic_torch.
    """
    if config.seed is None and not getattr(config, "deterministic", False):
        return

    seed_value = config.seed if config.seed is not None else 0
    random.seed(seed_value)
    np.random.seed(seed_value)

    torch = _load_torch()
    if torch is None:
        return

    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    if getattr(config, "deterministic_torch", False) and hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
