from __future__ import annotations

import inspect
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel
from src.core.channel import Receiver, Sender
from src.core.channel import UIReceiver, UISender

if TYPE_CHECKING:
    from src.core.graph import Graph, GraphManager


class Status(Enum):
    STARTUP = "startup"
    SETUP = "setup"
    RUNNING = "running"
    STOPPED = "stopped"


class Component[I: tuple[Receiver[Any] | None, ...], O: tuple[Sender[Any] | None, ...]](
    ABC
):
    def __init__(self) -> None:
        self._status = Status.STARTUP
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    @abstractmethod
    def type_(self) -> str: ...

    @property
    def status(self) -> Status:
        return self._status

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def setup(self) -> None:
        """Override to perform heavy initialization (e.g. model loading) before
        channels are wired.  Runs on the component thread before ``run()``."""

    @abstractmethod
    def run(self, inputs: I, outputs: O) -> None: ...

    def _safe_run(self, inputs: I, outputs: O) -> None:
        try:
            self._status = Status.SETUP
            self.setup()
            self._status = Status.RUNNING
            self.run(inputs, outputs)
        finally:
            self._status = Status.STOPPED

    def start(self, inputs: I, outputs: O) -> None:
        if self.status == Status.RUNNING:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._safe_run, args=(inputs, outputs), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """
        Idempotent.
        When a component instance is stopped, it will stop the running thread cooperatively by setting self._stop_event
        and expect the run() method to return. Input streams are unregistered from input channels as streams
        will periodically check and raise StopIteration when self._stop_event is set.
        """
        if self.status == Status.STOPPED:
            return
        self._stop_event.set()

    @classmethod
    def get_init_types(cls) -> dict[str, type]:
        """Returns {param_name: type} from __init__, excluding self."""
        hints = get_type_hints(cls.__init__)
        hints.pop("return", None)
        return hints

    @classmethod
    def _get_type_param(cls, index: int) -> type | None:
        for base in getattr(cls, "__orig_bases__", ()):
            origin = get_origin(base)
            if isinstance(origin, type) and issubclass(origin, Component):
                args = get_args(base)
                if len(args) > index:
                    return args[index]
        return None

    @classmethod
    def _resolve_tuple_types(cls, tp: type | None) -> dict[str, type]:
        if tp is None:
            return {}
        origin = get_origin(tp)
        # Parameterized generic NamedTuple, e.g. PassthroughInputs[T]
        if origin is not None and hasattr(origin, "_fields"):
            # Build substitution map: TypeVar -> actual arg
            params = getattr(origin, "__type_params__", ())
            args = get_args(tp)
            # Include type params in localns so get_type_hints can resolve them
            localns = dict(zip((p.__name__ for p in params), args))
            hints = get_type_hints(origin, localns=localns)
            sub = dict(zip(params, args))

            def _subst(t: type) -> type:
                if t in sub:
                    return sub[t]
                t_origin = get_origin(t)
                t_args = get_args(t)
                if t_origin and t_args:
                    return t_origin[tuple(_subst(a) for a in t_args)]
                return t

            return {name: _subst(hints[name]) for name in origin._fields}
        if hasattr(tp, "_fields"):
            hints = get_type_hints(tp)
            return {name: hints[name] for name in tp._fields}
        args = get_args(tp)
        if not args or args == ((),):
            return {}
        return {str(i + 1): arg for i, arg in enumerate(args)}

    @staticmethod
    def _is_ui_sender(t: type) -> bool:
        origin = get_origin(t) or t
        return isinstance(origin, type) and issubclass(origin, UISender)

    @staticmethod
    def _is_ui_receiver(t: type) -> bool:
        origin = get_origin(t) or t
        return isinstance(origin, type) and issubclass(origin, UIReceiver)

    @classmethod
    def get_input_types(cls) -> dict[str, type]:
        return {
            k: v
            for k, v in cls._resolve_tuple_types(cls._get_type_param(0)).items()
            if not cls._is_ui_receiver(v)
        }

    @classmethod
    def get_output_types(cls) -> dict[str, type]:
        return {
            k: v
            for k, v in cls._resolve_tuple_types(cls._get_type_param(1)).items()
            if not cls._is_ui_sender(v)
        }

    @classmethod
    def get_ui_input_types(cls) -> dict[str, type]:
        return {
            k: v
            for k, v in cls._resolve_tuple_types(cls._get_type_param(0)).items()
            if cls._is_ui_receiver(v)
        }

    @classmethod
    def get_ui_output_types(cls) -> dict[str, type]:
        return {
            k: v
            for k, v in cls._resolve_tuple_types(cls._get_type_param(1)).items()
            if cls._is_ui_sender(v)
        }

    @classmethod
    def get_config_options(
        cls, field: str, values: dict[str, Any] | None = None
    ) -> list[dict[str, str]] | None:
        """Override to provide runtime options for a config field.

        Args:
            field: the config field name (e.g. "config.source")
            values: current form values for dependent dropdowns

        Returns [{\"value\": \"...\", \"label\": \"...\"}, ...] or None.
        """
        return None

    @classmethod
    def from_args(cls, init_args: dict[str, Any]) -> Component[Any, Any]:
        """Construct a component instance, deserializing any BaseModel init params."""

        def _extract_basemodel_type(t: Any) -> type[BaseModel] | None:
            if isinstance(t, type) and issubclass(t, BaseModel):
                return t
            origin = get_origin(t)
            if origin is None:
                return None
            for arg in get_args(t):
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    return arg
            return None

        kwargs: dict[str, Any] = {}
        for k, v in cls.get_init_types().items():
            if k not in init_args:
                continue
            model_type = _extract_basemodel_type(v)
            if model_type is not None:
                raw = init_args[k]
                if isinstance(raw, BaseModel):
                    kwargs[k] = raw
                elif isinstance(raw, dict):
                    kwargs[k] = model_type(**raw)
                else:
                    kwargs[k] = model_type.model_validate(raw)
            else:
                kwargs[k] = init_args[k]
        return cls(**kwargs) if kwargs else cls()  # type: ignore[call-arg]

    @classmethod
    def registered_subclasses(cls) -> dict[str, type[Component[Any, Any]]]:
        """Returns all concrete subclasses as {name: class}, walking the full hierarchy."""
        from src.core import source, sink, conduit  # noqa: F401

        result: dict[str, type[Component[Any, Any]]] = {}

        def walk(subclass: type[Component[Any, Any]]) -> None:
            if not inspect.isabstract(subclass) and getattr(
                subclass, "_registerable", True
            ):
                result[subclass.__name__] = subclass
            for child in subclass.__subclasses__():
                walk(child)

        for child in cls.__subclasses__():
            walk(child)

        return result


class PrimitiveComponent[
    I: tuple[Receiver[Any] | None, ...],
    O: tuple[Sender[Any] | None, ...],
](
    Component[I, O],
    ABC,
):
    """A primitive morphism: a single concrete component with a threaded run loop."""

    @property
    def type_(self) -> str:
        return type(self).__name__


class CompositeComponent(Component[Any, Any]):
    """A composite morphism: its interface is derived from unmatched ports in the subgraph."""

    _registerable = False

    def __init__(self, type_: str, sub_graph: Graph) -> None:
        super().__init__()
        self._type = type_
        self._sub_graph = sub_graph
        self._inner_manager: GraphManager | None = None
        self._ext_inputs, self._ext_outputs = self._compute_boundary()

    @property
    def type_(self) -> str:
        return self._type

    def run(self, inputs: Any, outputs: Any) -> None:
        pass  # not used — start() is overridden

    def _compute_boundary(
        self,
    ) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
        """Find unmatched ports in the subgraph.

        Returns:
            ext_inputs: {ext_name: (node_id, slot)} for input slots with no internal source
            ext_outputs: {ext_name: (node_id, slot)} for output slots with no internal sink
        """
        connected_inputs: set[tuple[str, str]] = set()
        connected_outputs: set[tuple[str, str]] = set()
        for edge in self._sub_graph.edges:
            connected_inputs.add((edge.target_node, edge.target_slot))
            connected_outputs.add((edge.source_node, edge.source_slot))

        classes = Component.registered_subclasses()
        ext_inputs: dict[str, tuple[str, str]] = {}
        ext_outputs: dict[str, tuple[str, str]] = {}
        for node_id, node in self._sub_graph.nodes.items():
            cls = classes.get(node.type)
            if cls is None:
                continue
            for slot in cls.get_input_types():
                if (node_id, slot) not in connected_inputs:
                    ext_inputs[f"{node_id}.{slot}"] = (node_id, slot)
            for slot in cls.get_output_types():
                if (node_id, slot) not in connected_outputs:
                    ext_outputs[f"{node_id}.{slot}"] = (node_id, slot)
        return ext_inputs, ext_outputs

    def get_input_types(self) -> dict[str, type]:  # type: ignore[override]
        """Instance method override — returns types of unmatched input ports."""
        classes = Component.registered_subclasses()
        result: dict[str, type] = {}
        for ext_name, (node_id, slot) in self._ext_inputs.items():
            node = self._sub_graph.nodes[node_id]
            cls = classes.get(node.type)
            if cls is not None:
                slot_types = cls.get_input_types()
                if slot in slot_types:
                    result[ext_name] = slot_types[slot]
        return result

    def get_output_types(self) -> dict[str, type]:  # type: ignore[override]
        """Instance method override — returns types of unmatched output ports."""
        classes = Component.registered_subclasses()
        result: dict[str, type] = {}
        for ext_name, (node_id, slot) in self._ext_outputs.items():
            node = self._sub_graph.nodes[node_id]
            cls = classes.get(node.type)
            if cls is not None:
                slot_types = cls.get_output_types()
                if slot in slot_types:
                    result[ext_name] = slot_types[slot]
        return result

    def start(self, inputs: Any, outputs: Any) -> None:
        """Create inner GraphManager, patch boundary handles, and run."""
        if self.status == Status.RUNNING:
            return
        self._stop_event.clear()
        self._status = Status.SETUP

        from src.core.graph import GraphManager

        self._inner_manager = GraphManager(self._sub_graph)

        # Patch boundary: inject outer handles into inner manager's handle dicts.
        for i, (ext_name, (node_id, slot)) in enumerate(self._ext_inputs.items()):
            if hasattr(inputs, "_fields"):
                outer_receiver = getattr(inputs, ext_name, None)
            else:
                outer_receiver = inputs[i] if i < len(inputs) else None
            if outer_receiver is not None:
                self._inner_manager._receiver_handles[(node_id, slot)] = outer_receiver

        for i, (ext_name, (node_id, slot)) in enumerate(self._ext_outputs.items()):
            if hasattr(outputs, "_fields"):
                outer_sender = getattr(outputs, ext_name, None)
            else:
                outer_sender = outputs[i] if i < len(outputs) else None
            if outer_sender is not None:
                self._inner_manager._sender_handles[(node_id, slot)] = outer_sender

        self._inner_manager.run()
        self._status = Status.RUNNING

    def stop(self) -> None:
        super().stop()
        if self._inner_manager is not None:
            self._inner_manager.stop()
