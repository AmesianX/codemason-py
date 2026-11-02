"""
Environment Utilities for CodeMason
Equivalent to src/utils/env.ts

NOTE: paths are NOT defined here anymore. The single source of truth is
config.py (CONFIG_DIR / MEMORY_DIR, derived from $CODEMASON_CONFIG_DIR).
This module only re-exports them (as str, for os.path-based callers) and
provides the ensure_* helpers.
"""

import os

from .config import CONFIG_DIR as _CONFIG_DIR, MEMORY_DIR as _MEMORY_DIR

# Re-export as plain strings for backward compat with os.path.join callers.
CONFIG_DIR = str(_CONFIG_DIR)
MEMORY_DIR = str(_MEMORY_DIR)


def ensure_memory_dir() -> str:
    """Ensure memory directory exists and return its path"""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    return MEMORY_DIR


def ensure_config_dir() -> str:
    """Ensure config directory exists and return its path"""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    return CONFIG_DIR
