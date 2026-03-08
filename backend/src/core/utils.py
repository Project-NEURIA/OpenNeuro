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


def resize_and_crop(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize to cover target dimensions, then center-crop to exact size."""
    import cv2

    h_src, w_src = frame.shape[:2]
    scale = max(width / w_src, height / h_src)
    w_s = int(round(w_src * scale))
    h_s = int(round(h_src * scale))
    resized = cv2.resize(frame, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
    y0 = (h_s - height) // 2
    x0 = (w_s - width) // 2
    return resized[y0 : y0 + height, x0 : x0 + width]
