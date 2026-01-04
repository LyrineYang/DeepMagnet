from pathlib import Path
from typing import Optional

import numpy as np


def render_volume(mask: np.ndarray, bfield: Optional[np.ndarray] = None, out_path: Optional[str | Path] = None) -> None:
    """
    Quick volume rendering helper using PyVista if available.
    mask: (D,H,W) array, bfield optional (3,D,H,W)
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed; skipping render.")
        return
    grid = pv.wrap(mask.astype(np.float32))
    p = pv.Plotter(off_screen=out_path is not None)
    opacity = [0, 0.05, 0.1, 0.3, 0.6, 0.9, 1]
    p.add_volume(grid, cmap="viridis", opacity=opacity)
    if bfield is not None:
        # Downsample vectors for clarity.
        bf = np.transpose(bfield, (1, 2, 3, 0))
        sl = slice(None, None, max(1, bf.shape[0] // 10))
        vec = bf[sl, sl, sl]
        xx, yy, zz = np.mgrid[0 : vec.shape[0], 0 : vec.shape[1], 0 : vec.shape[2]]
        pts = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        arrows = pv.PolyData(pts)
        arrows["vectors"] = vec.reshape(-1, 3)
        p.add_arrows(pts, arrows["vectors"], mag=0.5, color="orange")
    p.show(screenshot=str(out_path) if out_path else None)
