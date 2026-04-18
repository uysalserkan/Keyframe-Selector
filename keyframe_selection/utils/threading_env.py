"""
CPU thread configuration for OpenCV, PyTorch CPU ops, and math libraries.

Call ``configure_host_threading`` once at pipeline start so sequential stages
use all cores; parallel worker pools should temporarily reduce OpenCV threads
to avoid oversubscription.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_configured_n: Optional[int] = None


def resolve_num_threads(config_value: Optional[int] = None) -> int:
    """
    Effective CPU thread count for host libraries and worker pools.

    Priority: explicit config > KEYFRAME_NUM_THREADS env > os.cpu_count().
    """
    if config_value is not None and config_value > 0:
        return int(config_value)
    env_raw = os.environ.get("KEYFRAME_NUM_THREADS", "").strip()
    if env_raw.isdigit() and int(env_raw) > 0:
        return int(env_raw)
    return max(1, (os.cpu_count() or 1))


def configure_host_threading(num_threads: Optional[int] = None) -> int:
    """
    Apply thread settings to OpenCV, PyTorch (CPU), and common BLAS env vars.

    Returns the resolved thread count used.
    """
    global _configured_n
    n = resolve_num_threads(num_threads)
    _configured_n = n

    # Hint BLAS/OpenMP stacks if the user did not already set them
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "MKL_DOMAIN_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        if key not in os.environ:
            os.environ[key] = str(n)

    try:
        import cv2

        cv2.setNumThreads(n)
    except Exception as e:
        logger.debug("OpenCV setNumThreads skipped: %s", e)

    try:
        import torch

        torch.set_num_threads(n)
        # Inter-op: keep small to limit oversubscription with DataLoader-style work
        inter = max(1, min(8, max(1, n // 4)))
        try:
            torch.set_num_interop_threads(inter)
        except RuntimeError:
            pass
    except Exception as e:
        logger.debug("PyTorch thread config skipped: %s", e)

    logger.info("Host threading: num_threads=%s (OpenCV/PyTorch CPU; BLAS env hinted)", n)
    return n


def get_last_configured_num_threads() -> int:
    """Return last value passed to configure_host_threading, or resolve default."""
    if _configured_n is not None:
        return _configured_n
    return resolve_num_threads(None)
