"""Pytest config: set Sage env vars + src/ path before any plane import."""

import os
import sys

# Repo root, regardless of where pytest is run from.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, os.path.join(ROOT, "src"))

# Sage writes to DOT_SAGE / SAGE_CACHE_DIR; point them at a writable local dir.
d_sage = os.path.join(ROOT, ".sage")
os.makedirs(d_sage, exist_ok=True)
os.environ.setdefault("DOT_SAGE", d_sage)
os.environ.setdefault("SAGE_CACHE_DIR", d_sage)
