from __future__ import annotations

import collections
import itertools
import re
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, overload

if TYPE_CHECKING:
    from src.core.channel import Receiver
    from src.core.component import ThreadedComponent
    from src.core.frames import Frame

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


@overload
def drain[F1: Frame](
    subscriber: ThreadedComponent,
    r1: Receiver[F1] | None,
    /,
) -> Iterator[tuple[F1 | None]]: ...
@overload
def drain[F1: Frame, F2: Frame](
    subscriber: ThreadedComponent,
    r1: Receiver[F1] | None,
    r2: Receiver[F2] | None,
    /,
) -> Iterator[tuple[F1 | None, F2 | None]]: ...
@overload
def drain[F1: Frame, F2: Frame, F3: Frame](
    subscriber: ThreadedComponent,
    r1: Receiver[F1] | None,
    r2: Receiver[F2] | None,
    r3: Receiver[F3] | None,
    /,
) -> Iterator[tuple[F1 | None, F2 | None, F3 | None]]: ...
@overload
def drain[F1: Frame, F2: Frame, F3: Frame, F4: Frame](
    subscriber: ThreadedComponent,
    r1: Receiver[F1] | None,
    r2: Receiver[F2] | None,
    r3: Receiver[F3] | None,
    r4: Receiver[F4] | None,
    /,
) -> Iterator[tuple[F1 | None, F2 | None, F3 | None, F4 | None]]: ...
@overload
def drain[F1: Frame, F2: Frame, F3: Frame, F4: Frame, F5: Frame](
    subscriber: ThreadedComponent,
    r1: Receiver[F1] | None,
    r2: Receiver[F2] | None,
    r3: Receiver[F3] | None,
    r4: Receiver[F4] | None,
    r5: Receiver[F5] | None,
    /,
) -> Iterator[tuple[F1 | None, F2 | None, F3 | None, F4 | None, F5 | None]]: ...
@overload
def drain[F1: Frame, F2: Frame, F3: Frame, F4: Frame, F5: Frame, F6: Frame](
    subscriber: ThreadedComponent,
    r1: Receiver[F1] | None,
    r2: Receiver[F2] | None,
    r3: Receiver[F3] | None,
    r4: Receiver[F4] | None,
    r5: Receiver[F5] | None,
    r6: Receiver[F6] | None,
    /,
) -> Iterator[tuple[F1 | None, F2 | None, F3 | None, F4 | None, F5 | None, F6 | None]]: ...
def drain(
    subscriber: ThreadedComponent, /, *receivers: Receiver[Any] | None
) -> Iterator[tuple[Any, ...]]:
    """Drain receivers (no_block) and yield one frame at a time, ordered by pts.

    Each yielded tuple has exactly one non-None value at the index of
    its source receiver, so callers can match frames to inputs.
    None receivers are treated as unconnected (skipped).
    """
    n = len(receivers)
    pending: list[tuple[int, Any]] = []
    for i, recv in enumerate(receivers):
        if recv is None:
            continue
        for frame in recv(subscriber, no_block=True):
            if frame is None:
                break
            pending.append((i, frame))

    if not pending:
        return

    pending.sort(key=lambda x: x[1].pts)
    for idx, frame in pending:
        row: list[Any] = [None] * n
        row[idx] = frame
        yield tuple(row)


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


def cut_space(buf: str) -> int:
    return max(buf.rfind(" "), buf.rfind("\n"), buf.rfind("\t"))


_SENT_END = re.compile(r"""[.!?…]+["'\)\]\}]*($|\s)""")


def cut_sentence(buf: str) -> int:
    last = -1
    for m in _SENT_END.finditer(buf):
        cut = m.end() - 1
        while cut >= 0 and buf[cut].isspace():
            cut -= 1
        last = max(last, cut)
    return last


@dataclass
class StreamFilter:
    speak_buf: str = ""
    in_square: int = 0
    in_paren: int = 0
    in_angle: int = 0
    in_bold: bool = False
    in_italic: bool = False
    cut_fn: Callable[[str], int] = field(default=cut_sentence)

    def feed(self, token: str, force: bool = False) -> str:
        self._consume(token)
        if force:
            out, self.speak_buf = self.speak_buf, ""
            return out

        cut = self.cut_fn(self.speak_buf)
        if cut < 0:
            return ""
        out = self.speak_buf[: cut + 1]
        self.speak_buf = self.speak_buf[cut + 1 :]
        return out

    def _consume(self, token: str) -> None:
        i = 0
        while i < len(token):
            ch = token[i]
            if ch == "*" and i + 1 < len(token) and token[i + 1] == "*":
                self.in_bold = not self.in_bold
                i += 2
                continue
            if ch == "*":
                self.in_italic = not self.in_italic
                i += 1
                continue
            if not (self.in_bold or self.in_italic):
                if ch == "[":
                    self.in_square += 1
                    i += 1
                    continue
                if ch == "]" and self.in_square:
                    self.in_square -= 1
                    i += 1
                    continue
                if ch == "(":
                    self.in_paren += 1
                    i += 1
                    continue
                if ch == ")" and self.in_paren:
                    self.in_paren -= 1
                    i += 1
                    continue
                if ch == "<":
                    self.in_angle += 1
                    i += 1
                    continue
                if ch == ">" and self.in_angle:
                    self.in_angle -= 1
                    i += 1
                    continue
            if (
                self.in_square
                or self.in_paren
                or self.in_angle
                or self.in_bold
                or self.in_italic
            ):
                i += 1
                continue
            self.speak_buf += ch
            i += 1


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
