"""Auto-setup utility for Hero environment variables."""
import os
import shutil
from pathlib import Path

def ensure_hero_env_file():
    """Create set_hero_env_vars.py from template if it doesn't exist."""
    hero_utils_dir = Path(__file__).parent
    target_file = hero_utils_dir / "set_hero_env_vars.py"
    template_file = hero_utils_dir / "set_hero_env_vars_template.py"
    
    if not target_file.exists() and template_file.exists():
        shutil.copy2(template_file, target_file)
        print(f"Created {target_file} from template.")
        print("Edit this file with your Hero credentials if using Hero features.")
        return True
    return False

if __name__ == "__main__":
    ensure_hero_env_file()