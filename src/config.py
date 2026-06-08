from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = PROJECT_ROOT / "archive"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ALIGNED_DIR = OUTPUT_DIR / "aligned"
FEATURES_DIR = OUTPUT_DIR / "features"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"

VIDEO_FPS_DEFAULT = 30.0
ML_RESAMPLE_HZ = 1.0
VIDEO_RESAMPLE_HZ = 30.0
HR_BAND = (0.7, 4.0)

RANDOM_STATE = 42
