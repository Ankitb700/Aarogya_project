import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "")

DEFAULT_MODEL = "FacePhys.rlap"

_rppg = None

def get_rppg():
    global _rppg
    if _rppg is not None:
        return _rppg
    try:
        import cv2
    except (ImportError, OSError):
        import subprocess
        sys.modules.pop("cv2", None)
        for pkg in ("opencv-python", "opencv-contrib-python"):
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", pkg, "-y"],
                capture_output=True,
            )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "opencv-python-headless", "-q"]
        )
        import cv2
    import rppg
    _rppg = rppg
    return _rppg

def get_extract_vitals():
    from vitals_analyzer import extract_vitals
    return extract_vitals

__all__ = ["get_rppg", "get_extract_vitals", "DEFAULT_MODEL"]
