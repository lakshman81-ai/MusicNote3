from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import List, Optional


class CommandRunner:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, argv: List[str], cwd: Optional[str] = None, check: bool = True, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        start = time.time()
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=check)
        duration = time.time() - start
        self._append_log(argv, proc.returncode, duration, proc.stdout, proc.stderr)
        return proc

    def _append_log(self, argv: List[str], returncode: int, duration: float, stdout: str, stderr: str) -> None:
        record = {
            "argv": argv,
            "returncode": returncode,
            "duration": duration,
            "stdout": stdout[-4096:],
            "stderr": stderr[-4096:],
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

