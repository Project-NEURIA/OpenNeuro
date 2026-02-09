from pydantic import BaseModel, ConfigDict


class Graph[T](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    edges: dict[str, list[str]]
    nodes: dict[str, T]
