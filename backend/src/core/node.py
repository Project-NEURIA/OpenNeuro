import threading
from typing import Never, overload

from .topic import Topic

_NONE: Topic[Never] = Topic()


class Node[T1, T2 = Never, T3 = Never, T4 = Never]:
    @overload
    def __init__(self, c1: Topic[T1], /) -> None: ...
    @overload
    def __init__(self, c1: Topic[T1], c2: Topic[T2], /) -> None: ...
    @overload
    def __init__(self, c1: Topic[T1], c2: Topic[T2], c3: Topic[T3], /) -> None: ...
    @overload
    def __init__(self, c1: Topic[T1], c2: Topic[T2], c3: Topic[T3], c4: Topic[T4], /) -> None: ...

    def __init__(self, *topics: Topic) -> None:  # type: ignore[override]
        padded = topics + (_NONE,) * (4 - len(topics))
        self._topics: tuple[Topic[T1], Topic[T2], Topic[T3], Topic[T4]] = padded  # type: ignore[assignment]

    def get_topics(self) -> tuple[Topic[T1], Topic[T2], Topic[T3], Topic[T4]]:
        return self._topics

    def run(self) -> None:
        raise NotImplementedError

    def start(self) -> None:
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
