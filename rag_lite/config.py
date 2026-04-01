import os
from pathlib import Path

PROJECT_ROOT = Path(os.getcwd())
MODELS_CACHE_DIR = PROJECT_ROOT / "models_cache"

os.makedirs(MODELS_CACHE_DIR, exist_ok=True)