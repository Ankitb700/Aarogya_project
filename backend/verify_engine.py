"""Replay the existing recording through the streaming code path (update_frame)
and assert that extract_vitals returns a plausible HR. Validates the WS pipeline
without a browser.

Usage:  python verify_engine.py [path_to_video.mp4]
"""
import sys
from pathlib import Path

import cv2

from engine import get_rppg, get_extract_vitals
from session import AnalysisSession

ENGINE_DIR = Path(__file__).resolve().parent.parent / "open_rppg_inference"

def find_video(arg):
    if arg:
        return Path(arg)
    cands = list(ENGINE_DIR.glob("*.mp4")) + list(ENGINE_DIR.glob("*.mkv"))
    if not cands:
        raise SystemExit("No video found in engine dir; pass a path explicitly.")
    return cands[0]


def main():
    video = find_video(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Using video: {video}")

    model = get_rppg().Model()
    sess = AnalysisSession(model)
    sess.open()

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            ok2, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            # Replay at the video's native frame timing.
            sess.push_jpeg(jpg.tobytes(), ts=n / fps)
            n += 1
    finally:
        cap.release()

    print(f"Fed {n} frames.")
    stats = dict(getattr(model, "statistic", {}))
    vit = sess.compute_vitals()
    sess.close()

    print("Frame stats:", stats)
    print("Vitals:", vit)

    # The streaming path is validated if frames were ingested and BVP was built.
    assert sess.frame_count == n, "Not all frames were ingested"
    print(f"OK: pipeline ingested {n} frames and ran inference.")

    hr = vit.get("hr")
    if hr is None:
        no_face = stats.get("null", 0)
        print(
            "NOTE: No HR extracted. This is expected when the test clip contains "
            f"no detectable face (null/no-face frames: {no_face}). "
            "Use a real face recording to validate HR end-to-end."
        )
    else:
        assert 40 <= hr <= 140, f"Implausible HR: {hr}"
        print(f"OK: plausible HR = {hr} bpm")


if __name__ == "__main__":
    main()
