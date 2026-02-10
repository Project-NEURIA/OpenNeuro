from pydantic import BaseModel, ConfigDict


class Edge(BaseModel):
    source_node: str
    source_slot: int
    target_node: str
    target_slot: int


class Graph[T](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    edges: list[Edge]
    nodes: dict[str, T]
