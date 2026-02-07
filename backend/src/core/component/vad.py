from ..base import Base


class VAD(Base[bytes]):
    def run(self) -> None:
        raise NotImplementedError
