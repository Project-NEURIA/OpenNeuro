from __future__ import annotations

import collections
import itertools
import threading

import numpy as np

_COUNTS: collections.defaultdict[str, itertools.count[int]] = collections.defaultdict(
    itertools.count
)
_COUNTS_LOCK = threading.Lock()
_ID = itertools.count()
_ID_LOCK = threading.Lock()


def obj_id() -> int:
    """Generate a unique id for an object.

    Returns:
        A unique integer identifier that increments globally across all objects.
    """
    with _ID_LOCK:
        return next(_ID)


def obj_count(obj) -> int:
    """Generate a unique count for an object based on its class.

    Args:
        obj: The object instance to count.

    Returns:
        A unique integer count that increments per class type.
    """
    with _COUNTS_LOCK:
        return next(_COUNTS[obj.__class__.__name__])


def auto_device(device: str = "auto"):  # type: ignore[return]
    """Select best available torch device: CUDA > MPS > CPU."""
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def auto_dtype(device):  # type: ignore[return]
    """Select best dtype for device. bfloat16 on CUDA, float32 elsewhere."""
    import torch

    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def center_crop_and_resize(frame: np.ndarray, target: int) -> np.ndarray:
    """Center-crop to largest square, resize to target x target."""
    import cv2

    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = frame[y0 : y0 + side, x0 : x0 + side]
    if side == target:
        return cropped
    return cv2.resize(cropped, (target, target), interpolation=cv2.INTER_AREA)
