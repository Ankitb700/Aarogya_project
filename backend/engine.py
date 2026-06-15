import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "")

try:
    import cv2
except (ImportError, OSError):
    import subprocess
    print("Installing opencv-python-headless...", file=sys.stderr)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "opencv-python-headless", "-q"]
    )
    import cv2

import rppg
from vitals_analyzer import extract_vitals

DEFAULT_MODEL = "FacePhys.rlap"

__all__ = ["rppg", "extract_vitals", "DEFAULT_MODEL"]
