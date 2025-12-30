from __future__ import annotations

import importlib
import shutil
import subprocess
from typing import Dict, List, Tuple

from .config_struct import BackendPolicyConfig, BackendPriorityConfig, ToolsConfig
from .models import BackendResolution


def _command_available(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _command_healthy(cmd: str, args: List[str]) -> bool:
    if not _command_available(cmd):
        return False
    try:
        subprocess.run([cmd] + args, capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def _import_available(pkg: str) -> Tuple[bool, bool]:
    try:
        importlib.import_module(pkg)
        return True, True
    except ModuleNotFoundError:
        return False, False
    except Exception:
        return True, False


class BackendResolver:
    """
    Resolves the best backend per feature based on availability, health, and policy.
    """

    def __init__(self, policy: BackendPolicyConfig, priority: BackendPriorityConfig, tools: ToolsConfig):
        self.policy = policy
        self.priority = priority
        self.tools = tools
        self.trace: List[BackendResolution] = []
        self.caps: Dict[str, str] = {}

    def resolve(self) -> Dict[str, str]:
        resolutions = {
            "separation": self._resolve_feature("separation", self.priority.separation),
            "piano_tx": self._resolve_feature("piano_tx", self.priority.piano_tx),
            "vocal_tx": self._resolve_feature("vocal_tx", self.priority.vocal_tx),
            "beats": self._resolve_feature("beats", self.priority.beats),
            "quantize": self._resolve_feature("quantize", self.priority.quantize),
            "midi_parser": self._resolve_feature("midi_parser", self.priority.midi_parser),
            "duration_fix": self._resolve_feature("duration_fix", self.priority.duration_fix),
        }
        return resolutions

    def _resolve_feature(self, feature: str, candidates: List[str]) -> str:
        requested = candidates[0] if candidates else None
        chosen = "off"
        reason = "no candidates"
        available = False
        healthy = False

        for backend in candidates:
            available, healthy, caps = self._check_backend(feature, backend)
            if not available:
                reason = f"{backend} unavailable"
                continue
            if not healthy:
                reason = f"{backend} unhealthy"
                if self.policy.fallback_mode == "never":
                    chosen = backend
                    break
                continue
            chosen = backend
            reason = "selected"
            self.caps.update({f"{feature}.{backend}": caps})
            break
        else:
            if self.policy.fallback_mode == "strict" and not self.policy.allow_off:
                raise RuntimeError(f"Strict policy prevented selecting backend for {feature}")
            if not self.policy.allow_off and "none" in candidates:
                raise RuntimeError(f"Off/none disallowed for {feature}")

        self.trace.append(
            BackendResolution(
                feature=feature,
                requested=requested,
                resolved=chosen,
                available=available,
                healthy=healthy,
                reason=reason,
                caps={k: str(v) for k, v in (self.caps.items()) if k.startswith(f"{feature}.")},
            )
        )
        return chosen

    def _check_backend(self, feature: str, backend: str) -> Tuple[bool, bool, Dict[str, str]]:
        caps: Dict[str, str] = {}
        if backend in {"none", "off"}:
            return True, True, caps
        if backend == "demucs_cli":
            avail = _command_available(self.tools.demucs)
            healthy = _command_healthy(self.tools.demucs, ["--help"])
            return avail, healthy, caps
        if backend == "demucs_py":
            avail, healthy = _import_available("demucs")
            return avail, healthy, caps
        if backend in {"piano_cli", "mt3_cli"}:
            avail = _command_available("piano-transcriber")
            healthy = avail
            return avail, healthy, caps
        if backend == "bytedance_py":
            avail, healthy = _import_available("piano_transcription_inference")
            return avail, healthy, caps
        if backend in {"basic_pitch_py"}:
            avail, healthy = _import_available("basic_pitch")
            return avail, healthy, caps
        if backend == "basic_pitch_cli":
            avail = _command_available("basic-pitch")
            healthy = avail
            return avail, healthy, caps
        if backend == "sonic_cli":
            avail = _command_available(self.tools.sonic_annotator)
            healthy = _command_healthy(self.tools.sonic_annotator, ["-h"])
            return avail, healthy, caps
        if backend == "madmom_py":
            avail, healthy = _import_available("madmom")
            return avail, healthy, caps
        if backend == "music21_py":
            avail, healthy = _import_available("music21")
            return avail, healthy, caps
        if backend == "dp_cost_lite":
            return True, True, caps
        if backend == "grid_lite":
            return True, True, caps
        if backend == "mido_py":
            avail, healthy = _import_available("mido")
            return avail, healthy, caps
        if backend == "minimal":
            return True, True, caps
        if backend == "scipy_resample_poly":
            avail, healthy = _import_available("scipy.signal")
            return avail, healthy, caps
        if backend == "interp_linear":
            return True, True, caps
        return False, False, caps

