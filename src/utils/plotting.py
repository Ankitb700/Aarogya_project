from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_aligned_signals(df: pd.DataFrame, columns: list[str] | None = None):
    columns = columns or [c for c in ["bvp", "hr", "eda", "temp"] if c in df.columns]
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in columns:
        ax.plot(df["elapsed_seconds"], df[col], label=col, alpha=0.8)
    ax.set_xlabel("Elapsed seconds")
    ax.legend()
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_feature_histogram(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    df[column].hist(ax=ax, bins=min(20, max(3, len(df))))
    ax.set_title(column)
    ax.grid(True, alpha=0.25)
    return fig, ax
