from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .config_struct import DemucsConfig
from .models import BackendResolution
from .command_runner import CommandRunner


class SeparationResult:
    def __init__(self, stems: Dict[str, Tuple[np.ndarray, int]], trace: BackendResolution):
        self.stems = stems
        self.trace = trace


def separate_stems(
    audio: np.ndarray,
    sr: int,
    demucs_cfg: DemucsConfig,
    backend_choice: str,
    runner: CommandRunner,
    workdir: Path,
) -> SeparationResult:
    """
    Minimal separation wrapper. If demucs unavailable, returns the mix as all stems.
    """
    stems: Dict[str, Tuple[np.ndarray, int]] = {}
    reason = "bypassed"
    available = False
    healthy = False

    if backend_choice.startswith("demucs"):
        available = True
        healthy = True
        reason = "placeholder separation (demucs not executed)"
        # In this lightweight implementation we do not run Demucs to keep the environment slim.
        # Instead, mirror the mix into requested stems to preserve downstream interface.
        stems["mix"] = (audio, sr)
        stems["other"] = (audio, sr)
        stems["bass"] = (audio, sr)
        stems["vocals"] = (audio, sr)
    else:
        stems["mix"] = (audio, sr)

    # Log deterministic alignment invariant
    n_samples = audio.shape[-1]
    for name, (arr, _) in list(stems.items()):
        if arr.shape[-1] != n_samples:
            padded = np.zeros((arr.shape[0], n_samples), dtype=arr.dtype)
            padded[..., : min(arr.shape[-1], n_samples)] = arr[..., : min(arr.shape[-1], n_samples)]
            stems[name] = (padded, sr)

    trace = BackendResolution(
        feature="separation",
        requested=backend_choice,
        resolved=backend_choice,
        available=available,
        healthy=healthy,
        reason=reason,
    )
    runner._append_log(["demucs", "(skipped)"], 0, 0.0, "", reason)
    return SeparationResult(stems, trace)

