# backend/pipeline/utils_config.py
from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Sequence, Union

def coalesce_not_none(*vals: Any) -> Any:
    """Return the first value that is not None (0 is valid and must be preserved)."""
    for v in vals:
        if v is not None:
            return v
    return None

def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default

def safe_setattr(obj: Any, name: str, value: Any) -> bool:
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False

def apply_dotted_overrides(target: Any, overrides: Mapping[str, Any]) -> None:
    """
    Apply dotted-path overrides into nested dataclasses/objects/dicts.
    Creates intermediate dicts when needed.
    """
    for path, value in (overrides or {}).items():
        parts = str(path).split(".")
        cur = target
        for i, part in enumerate(parts):
            last = (i == len(parts) - 1)

            if isinstance(cur, dict):
                if last:
                    cur[part] = value
                    break
                nxt = cur.get(part, None)
                if nxt is None:
                    nxt = {}
                    cur[part] = nxt
                cur = nxt
                continue

            # object / dataclass
            if last:
                # if attribute exists -> set; else if cur has __dict__ -> set anyway
                try:
                    setattr(cur, part, value)
                except Exception:
                    # fallback: store into __dict__ if possible
                    d = getattr(cur, "__dict__", None)
                    if isinstance(d, dict):
                        d[part] = value
                break

            # intermediate: fetch existing
            nxt = None
            try:
                nxt = getattr(cur, part)
            except Exception:
                nxt = None

            # if missing, create dict container
            if nxt is None:
                nxt = {}
                try:
                    setattr(cur, part, nxt)
                except Exception:
                    d = getattr(cur, "__dict__", None)
                    if isinstance(d, dict):
                        d[part] = nxt

            cur = nxt
