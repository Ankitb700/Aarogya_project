from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal


def hrv_features(ibi_seconds: pd.Series | np.ndarray) -> dict[str, float]:
    ibi = np.asarray(ibi_seconds, dtype=float)
    ibi = ibi[np.isfinite(ibi)]
    if len(ibi) < 2:
        return {"rmssd": np.nan, "sdnn": np.nan, "pnn50": np.nan, "lf_hf": np.nan}
    diff = np.diff(ibi)
    rmssd = np.sqrt(np.mean(diff**2))
    sdnn = np.std(ibi, ddof=1) if len(ibi) > 1 else np.nan
    pnn50 = np.mean(np.abs(diff) > 0.05)
    lf_hf = np.nan
    if len(ibi) >= 8:
        times = np.cumsum(ibi)
        grid = np.arange(times[0], times[-1], 0.25)
        if len(grid) >= 8:
            interp = np.interp(grid, times, ibi)
            freqs, psd = signal.welch(interp, fs=4.0, nperseg=min(256, len(interp)))
            lf = psd[(freqs >= 0.04) & (freqs < 0.15)].sum()
            hf = psd[(freqs >= 0.15) & (freqs < 0.4)].sum()
            lf_hf = float(lf / hf) if hf > 0 else np.nan
    return {"rmssd": float(rmssd), "sdnn": float(sdnn), "pnn50": float(pnn50), "lf_hf": lf_hf}
