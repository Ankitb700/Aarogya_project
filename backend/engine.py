import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import rppg
from vitals_analyzer import extract_vitals

DEFAULT_MODEL = "FacePhys.rlap"

__all__ = ["rppg", "extract_vitals", "DEFAULT_MODEL"]
