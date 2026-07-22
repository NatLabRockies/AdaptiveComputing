"""
env_utils.py — Load KEY=value pairs from a plain-text file into os.environ.

Usage::

    from adaptive_computing.utils import load_env_file
    load_env_file("/path/to/env_vars.txt")

File format::

    # Lines beginning with # are ignored
    LITELLM_MODEL=gpt-4o
    LITELLM_API_KEY='sk-...'

Already-set environment variables are not overwritten, so values from the
calling process take precedence over values in the file.
"""

from __future__ import annotations

import os


def load_env_file(path: str) -> None:
    """Parse KEY=value lines from *path* and set them in :data:`os.environ`.

    - Lines starting with ``#`` and blank lines are skipped.
    - Values may be optionally quoted with single or double quotes.
    - Variables already present in the environment are not overwritten.

    Args:
        path: Absolute or relative path to the environment file.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"env file not found: {path}")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, raw_value = line.partition("=")
            key   = key.strip()
            value = raw_value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
