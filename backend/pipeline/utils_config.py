from __future__ import annotations
from typing import Any, Dict, Optional

def coalesce_not_none(*vals):
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

def apply_dotted_overrides(cfg: Any, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies dotted-path overrides to dataclasses/dicts.
    Returns dict of successfully applied overrides.
    Handles intermediate creation of dicts if missing or None.
    """
    applied: Dict[str, Any] = {}
    for path, val in (overrides or {}).items():
        try:
            cur = cfg
            parts = str(path).split(".")
            for k in parts[:-1]:
                if isinstance(cur, dict):
                    # Dict path
                    if k not in cur or cur[k] is None:
                        cur[k] = {}
                    cur = cur[k]
                else:
                    # Object/Dataclass path
                    # Check if attribute exists, if not or None, try to set as empty dict
                    if not hasattr(cur, k):
                        try:
                            setattr(cur, k, {})
                        except Exception:
                            # Cannot set attribute? Stop path
                            break

                    val_attr = getattr(cur, k)
                    if val_attr is None:
                        try:
                            setattr(cur, k, {})
                            val_attr = getattr(cur, k)
                        except Exception:
                             break
                    cur = val_attr

            last = parts[-1]
            if isinstance(cur, dict):
                cur[last] = val
            else:
                setattr(cur, last, val)
            applied[path] = val
        except Exception:
            continue
    return applied
