#!/usr/bin/env python3
"""
tools/l6_debug_runner.py

Wrapper for backend.tools.l6_debug_runner.
"""

import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.tools.l6_debug_runner import main

if __name__ == "__main__":
    sys.exit(main())
