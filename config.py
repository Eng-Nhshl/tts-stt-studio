"""Project-wide configuration and logging setup."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Silence overly noisy third-party loggers if necessary
for noisy in ("urllib3", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Paths / Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent  # repo root directory
DATASET_DIR = PROJECT_ROOT / "datasets"  # offensive-word JSONs live here
TEMP_DIR = Path(tempfile.gettempdir())  # OS temp dir – fallback scratch space
CACHE_DIR = PROJECT_ROOT / ".cache"  # deterministic TTS mp3 cache
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = PROJECT_ROOT / "outputs"  # user-visible artefacts
OUTPUT_TTS_DIR = OUTPUT_DIR / "tts"  # saved mp3 files
OUTPUT_STT_DIR = OUTPUT_DIR / "stt"  # saved transcript .txt files
for _p in (OUTPUT_DIR, OUTPUT_TTS_DIR, OUTPUT_STT_DIR):
    _p.mkdir(exist_ok=True)

__all__ = [
    "logging",
    "DATASET_DIR",
    "TEMP_DIR",
]
