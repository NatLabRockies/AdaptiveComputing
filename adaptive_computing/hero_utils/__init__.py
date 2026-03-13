"""Hero utilities for Adaptive Computing.

Auto-creates hero environment configuration on first import.
"""
from .setup import ensure_hero_env_file

# Auto-create the hero env file if it doesn't exist
ensure_hero_env_file()