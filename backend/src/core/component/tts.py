from ..base import Base


class TTS(Base[bytes]):
    def run(self) -> None:
        raise NotImplementedError
