from ..base import Base


class LLM(Base[str]):
    def run(self) -> None:
        raise NotImplementedError
