"""
load_env.py — load key=value pairs from env_vars.txt into os.environ.

Usage:
    from load_env import load_env_file
    load_env_file()   # loads agent/env_vars.txt by default
"""

import os

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ENV_FILE = os.path.join(_AGENT_DIR, "env_vars.txt")


def load_env_file(path=_DEFAULT_ENV_FILE):
    """
    Parse KEY='value' or KEY=value lines into os.environ.
    Lines starting with # and blank lines are ignored.
    Already-set variables are not overwritten.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("env file not found: {}".format(path))
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, raw_value = line.partition("=")
            key = key.strip()
            value = raw_value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
