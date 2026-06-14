"""Bridge to the existing open_rppg_inference engine.

Adds the engine directory to sys.path so `import rppg` and the
`extract_vitals` helper from `vitals_analyzer.py` resolve, regardless of the
directory uvicorn is launched from.
"""
import os
import sys
from pathlib import Path

# .../phase_1/backend/engine.py  ->  .../phase_1/open_rppg_inference
ENGINE_DIR = (Path(__file__).resolve().parent.parent / "open_rppg_inference")

if not ENGINE_DIR.exists():
    raise RuntimeError(f"open_rppg_inference not found at {ENGINE_DIR}")

if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

# Keep JAX from grabbing all GPU memory up front (mirrors the notebook setup).
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import rppg  # noqa: E402
from vitals_analyzer import extract_vitals  # noqa: E402

DEFAULT_MODEL = "FacePhys.rlap"

__all__ = ["rppg", "extract_vitals", "DEFAULT_MODEL", "ENGINE_DIR"]
