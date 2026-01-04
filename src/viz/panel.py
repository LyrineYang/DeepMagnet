from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .volume import render_volume


def plot_waveform(signals: np.ndarray, out_path: Optional[str | Path] = None, show: bool = False) -> None:
    plt.figure(figsize=(6, 3))
    plt.imshow(signals, aspect="auto", origin="lower", cmap="magma")
    plt.xlabel("Sample")
    plt.ylabel("Trajectory step")
    plt.title("Sensor waveform")
    plt.colorbar(label="Amplitude")
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def render_panel(
    signals: np.ndarray,
    mask: np.ndarray,
    bfield: Optional[np.ndarray] = None,
    traj: Optional[np.ndarray] = None,
    out_dir: str | Path = "outputs/demo",
    live: bool = False,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_waveform(signals, out_dir / "waveform.png", show=live)
    render_volume(mask, bfield, None if live else out_dir / "volume.png")
