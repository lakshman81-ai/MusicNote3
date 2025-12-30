from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Dict, Iterable, Tuple
import os
import importlib
import importlib.util

if importlib.util.find_spec("tomllib"):
    tomllib = importlib.import_module("tomllib")  # type: ignore
else:  # pragma: no cover
    tomllib = importlib.import_module("tomli")  # type: ignore

from .config_struct import Wav2XmlConfig


class UnknownConfigKeyError(ValueError):
    """Raised when a configuration key is not present in the schema."""


def _load_toml(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def cfg_get(obj: Any, dotted: str, default: Any = None) -> Any:
    """Safely access dynamic config keys using dotted syntax."""
    parts = dotted.split(".")
    cur = obj
    for part in parts:
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


class ConfigLoader:
    """
    Strict, provenance-tracking config loader.
    Unknown keys raise errors and every value is tagged with a source string.
    """

    def __init__(self) -> None:
        self.provenance: Dict[str, str] = {}
        self.layers: Dict[str, Dict[str, Any]] = {}

    def load(self, base_dir: str, preset: str | None = None, overrides: Iterable[Tuple[str, Any]] | None = None) -> Wav2XmlConfig:
        config = Wav2XmlConfig()

        default_path = os.path.join(base_dir, "default.toml")
        if os.path.exists(default_path):
            self._apply_layer(config, _load_toml(default_path), "default")

        if preset:
            preset_path = os.path.join(base_dir, "presets", f"{preset}.toml")
            if not os.path.exists(preset_path):
                raise FileNotFoundError(f"Preset '{preset}' not found at {preset_path}")
            self._apply_layer(config, _load_toml(preset_path), f"preset:{preset}")

        if overrides:
            override_dict: Dict[str, Any] = {}
            for dotted, value in overrides:
                override_dict.setdefault("__root__", {})[dotted] = value
            self._apply_overrides(config, override_dict["__root__"], "override")

        return config

    def _apply_overrides(self, config: Wav2XmlConfig, data: Dict[str, Any], source: str) -> None:
        for dotted, value in data.items():
            parts = dotted.split(".")
            target = config
            path_parts = []
            for part in parts[:-1]:
                path_parts.append(part)
                if not hasattr(target, part):
                    raise UnknownConfigKeyError(".".join(path_parts))
                target = getattr(target, part)
            leaf = parts[-1]
            path_parts.append(leaf)
            if not hasattr(target, leaf):
                raise UnknownConfigKeyError(".".join(path_parts))
            setattr(target, leaf, value)
            self.provenance[".".join(path_parts)] = source

    def _apply_layer(self, config: Wav2XmlConfig, data: Dict[str, Any], source: str, prefix: str = "") -> None:
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            if not hasattr(config, key):
                raise UnknownConfigKeyError(path)
            existing = getattr(config, key)
            if is_dataclass(existing):
                if not isinstance(value, dict):
                    raise TypeError(f"Expected mapping at {path}")
                self._apply_layer(existing, value, source, prefix=path)
            else:
                setattr(config, key, value)
                self.provenance[path] = source
