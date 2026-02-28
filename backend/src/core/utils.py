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


def to_numpy(t: object) -> np.ndarray:
    """Convert a torch tensor or array-like to a numpy array."""
    import torch

    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


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


def center_crop_and_resize(
    frame: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Center-crop to the largest rectangle matching the target aspect ratio, then resize."""
    import cv2

    h, w = frame.shape[:2]
    target_ratio = width / height
    src_ratio = w / h

    if src_ratio > target_ratio:
        crop_w = int(h * target_ratio)
        x0 = (w - crop_w) // 2
        cropped = frame[:, x0 : x0 + crop_w]
    else:
        crop_h = int(w / target_ratio)
        y0 = (h - crop_h) // 2
        cropped = frame[y0 : y0 + crop_h, :]

    if cropped.shape[:2] == (height, width):
        return cropped
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
