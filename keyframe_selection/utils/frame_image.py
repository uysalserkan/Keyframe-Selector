"""
Load frame pixels from memory or from disk (memory-efficient video sampling).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


def _ensure_bgr_uint8(arr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Ensure non-empty HxWx3 BGR for encoders and cv2.COLOR_BGR2GRAY callers."""
    if arr is None or arr.size == 0 or arr.ndim < 2:
        raise ValueError("Frame image is empty or invalid")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h < 1 or w < 1:
        raise ValueError("Frame image has zero width or height")
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    return arr[:, :, :3].copy() if arr.shape[2] > 3 else arr


def load_frame_bgr(frame: Any) -> NDArray[np.uint8]:
    """
    Return BGR uint8 image for a FrameData-like object.

    Uses in-memory ``image`` when set; otherwise reads ``path`` with OpenCV.
    Disk read uses IMREAD_COLOR so grayscale files become 3-channel BGR.
    """
    img = getattr(frame, "image", None)
    if img is not None:
        return _ensure_bgr_uint8(img)
    path = getattr(frame, "path", None)
    if path is None:
        raise ValueError("Frame has no in-memory image and no path")
    path = Path(path)
    data = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if data is None:
        raise OSError(f"Could not read frame image: {path}")
    return _ensure_bgr_uint8(data)
