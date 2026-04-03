from __future__ import annotations

import inspect
import threading
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin, get_type_hints

from pydantic import BaseModel
from src.core.channel import Receiver, Sender
from src.core.channel import UIReceiver, UISender
from src.core.log_capture import get_log_store

if TYPE_CHECKING:
    from src.core.graph import Graph, GraphManager


IOTag = Literal["source", "conduit", "sink"]
FunctionalityTag = Literal[
    "audio", "video", "llm", "image", "movement", "misc", "other"
]
GPUTag = Literal["cpu", "nvidia", "apple", "intel", "amd"]


class Tag(BaseModel):
    io: set[IOTag]
    functionality: set[FunctionalityTag]
    gpu: set[GPUTag] = {"cpu"}


class Status(Enum):
    STARTUP = "startup"
    SETUP = "setup"
    RUNNING = "running"
    STOPPED = "stopped"


# ---------------------------------------------------------------------------
# Component — abstract base for all components
# ---------------------------------------------------------------------------


class Component[
    I: tuple[Receiver[Any] | None, ...],
    O: tuple[Sender[Any] | None, ...],
](ABC):
    tags: Tag = Tag(io=set(), functionality=set())
    description: str = ""

    def __init__(self) -> None:
        self._status = Status.STARTUP

    @property
    @abstractmethod
    def type_(self) -> str: ...

    @property
    def status(self) -> Status:
        return self._status

    def setup(self, outputs: O) -> None:
        """Called synchronously by GraphManager before threads start.

        Override to perform initialization or emit initial values into outputs.
        """

    def start(self, inputs: I, outputs: O) -> None:
        """Start the component. Subclasses override with their execution model."""
        self._status = Status.RUNNING

    def stop(self) -> None:
        """Idempotent. Signals the component to stop."""
        self._status = Status.STOPPED

    # --- Reflection / introspection ---

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
            params = getattr(origin, "__type_params__", ())
            args = get_args(tp)
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

    @abstractmethod
    def get_input_types(self) -> dict[str, type]: ...

    @abstractmethod
    def get_output_types(self) -> dict[str, type]: ...

    @abstractmethod
    def get_ui_input_types(self) -> dict[str, type]: ...

    @abstractmethod
    def get_ui_output_types(self) -> dict[str, type]: ...

    @classmethod
    def get_options(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Return a nested dict of field → options for all fields that have dynamic options.

        Args:
            values: current form values, for dependent dropdowns where
                    field B's options depend on field A's selection.

        Top-level fields: {"path": [{"value": "...", "label": "..."}, ...]}
        Nested config fields: {"config": {"field": [{"value": "...", "label": "..."}, ...]}}
        """
        return {}

    @classmethod
    def from_args(cls, init_args: dict[str, Any]) -> Component[Any, Any]:
        """Construct a component instance, deserializing any BaseModel init params.

        Extra keys in *init_args* that do not correspond to any ``__init__``
        parameter are silently ignored.  Missing **required** parameters
        (those without a default value) raise a clear ``ValueError``.
        """

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

        # Determine which init params are required (no default value).
        sig = inspect.signature(cls.__init__)
        required_params: set[str] = set()
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.default is inspect.Parameter.empty:
                required_params.add(name)

        # Only consider params that __init__ actually declares.
        valid_params = cls.get_init_types()

        # Check for missing required fields before doing any work.
        missing = required_params - init_args.keys()
        if missing:
            raise ValueError(
                f"{cls.__name__}: missing required init field(s): "
                f"{', '.join(sorted(missing))}"
            )

        kwargs: dict[str, Any] = {}
        for k, v in valid_params.items():
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
                raw = init_args[k]
                # Convert string to Path if any part of the type hint is Path
                hint_types = get_args(v) if get_origin(v) is not None else (v,)
                is_path = any(
                    isinstance(t, type) and issubclass(t, Path) for t in hint_types
                )
                if is_path and isinstance(raw, str):
                    kwargs[k] = Path(raw)
                else:
                    kwargs[k] = raw
        return cls(**kwargs) if kwargs else cls()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# PrimitiveComponent — single component (not a subgraph)
# ---------------------------------------------------------------------------


class PrimitiveComponent[
    I: tuple[Receiver[Any] | None, ...],
    O: tuple[Sender[Any] | None, ...],
](Component[I, O], ABC):
    """A primitive morphism: a single component, not a composite.

    Not directly instantiable — use ThreadedComponent.
    """

    @property
    def type_(self) -> str:
        return type(self).__name__

    @classmethod
    def _class_input_types(cls) -> dict[str, type]:
        return {
            k: v
            for k, v in cls._resolve_tuple_types(cls._get_type_param(0)).items()
            if not cls._is_ui_receiver(v)
        }

    @classmethod
    def _class_output_types(cls) -> dict[str, type]:
        return {
            k: v
            for k, v in cls._resolve_tuple_types(cls._get_type_param(1)).items()
            if not cls._is_ui_sender(v)
        }

    @classmethod
    def _class_ui_input_types(cls) -> dict[str, type]:
        return {
            k: v
            for k, v in cls._resolve_tuple_types(cls._get_type_param(0)).items()
            if cls._is_ui_receiver(v)
        }

    @classmethod
    def _class_ui_output_types(cls) -> dict[str, type]:
        return {
            k: v
            for k, v in cls._resolve_tuple_types(cls._get_type_param(1)).items()
            if cls._is_ui_sender(v)
        }

    def get_input_types(self) -> dict[str, type]:
        return type(self)._class_input_types()

    def get_output_types(self) -> dict[str, type]:
        return type(self)._class_output_types()

    def get_ui_input_types(self) -> dict[str, type]:
        return type(self)._class_ui_input_types()

    def get_ui_output_types(self) -> dict[str, type]:
        return type(self)._class_ui_output_types()

    @classmethod
    def registered_subclasses(cls) -> dict[str, type[PrimitiveComponent[Any, Any]]]:
        """Returns all concrete subclasses as {name: class}, walking from PrimitiveComponent."""
        from src.core import source, sink, conduit  # noqa: F401

        result: dict[str, type[PrimitiveComponent[Any, Any]]] = {}

        skip = {
            PrimitiveComponent,
            ThreadedComponent,
        }

        def walk(subclass: type[PrimitiveComponent[Any, Any]]) -> None:
            if (
                subclass not in skip
                and not inspect.isabstract(subclass)
                and getattr(subclass, "_registerable", True)
            ):
                result[subclass.__name__] = subclass
            for child in subclass.__subclasses__():
                walk(child)  # type: ignore[arg-type]

        walk(PrimitiveComponent)
        return result


# ---------------------------------------------------------------------------
# ThreadedComponent — primitive with a threaded run loop
# ---------------------------------------------------------------------------


class ThreadedComponent[
    I: tuple[Receiver[Any] | None, ...],
    O: tuple[Sender[Any] | None, ...],
](PrimitiveComponent[I, O], ABC):
    """A primitive component that runs in a daemon thread."""

    def __init__(self) -> None:
        super().__init__()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    @abstractmethod
    def run(self, inputs: I, outputs: O) -> None: ...

    def _safe_run(self, inputs: I, outputs: O) -> None:
        try:
            self._status = Status.RUNNING
            self.run(inputs, outputs)
        except Exception:
            traceback.print_exc()
        finally:
            # Registration happens in GraphManager when start() is called.
            # Unregister here to ensure buffered partial lines are flushed.
            get_log_store().unregister_thread()
            self._status = Status.STOPPED

    def start(self, inputs: I, outputs: O) -> None:
        if self.status in (Status.RUNNING, Status.SETUP):
            return
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._safe_run, args=(inputs, outputs), daemon=True
        )
        self._thread.start()

    def get_ident(self) -> int | None:
        return None if self._thread is None else self._thread.ident

    def stop(self) -> None:
        if self.status == Status.STOPPED:
            return
        self._stop_event.set()
        # destruct is called in _safe_run's finally block (in the thread)

    def join(self, timeout: float = 5.0) -> None:
        """Block until the component's thread finishes or *timeout* expires."""
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# EmitOnStart — mixin for components that send values synchronously on start
# ---------------------------------------------------------------------------


class EmitOnStart[E: tuple[Sender[Any] | None, ...]](ABC):
    """Mixin for components that need to send values before threads start.

    GraphManager calls emit(outputs) synchronously during run(),
    before start() is called. This guarantees values are in channels
    before any downstream component's thread reads them.
    """

    @abstractmethod
    def emit(self, outputs: E) -> None: ...


# ---------------------------------------------------------------------------
# CompositeComponent — subgraph wrapper
# ---------------------------------------------------------------------------


class CompositeComponent(Component[Any, Any]):
    """A composite morphism: its interface is derived from unmatched ports in the subgraph."""

    _registerable = False

    def __init__(
        self,
        type_: str,
        sub_graph: Graph,
        tags: Tag = Tag(io={"conduit"}, functionality={"misc"}),
        description: str = "",
    ) -> None:
        super().__init__()
        self._type = type_
        self.tags = tags
        self.description = description
        self._sub_graph = sub_graph
        self._inner_manager: GraphManager | None = None
        self._ext_inputs, self._ext_outputs = self._compute_boundary()

    @property
    def type_(self) -> str:
        return self._type

    def _compute_boundary(
        self,
    ) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
        connected_inputs: set[tuple[str, str]] = set()
        connected_outputs: set[tuple[str, str]] = set()
        for edge in self._sub_graph.edges:
            connected_inputs.add((edge.target_node, edge.target_slot))
            connected_outputs.add((edge.source_node, edge.source_slot))

        classes = PrimitiveComponent.registered_subclasses()
        ext_inputs: dict[str, tuple[str, str]] = {}
        ext_outputs: dict[str, tuple[str, str]] = {}
        for node_id, node in self._sub_graph.nodes.items():
            cls = classes.get(node.type)
            if cls is None:
                # Node is a CompositeComponent — nested composites not yet supported
                continue
            for slot in cls._class_input_types():
                if (node_id, slot) not in connected_inputs:
                    ext_inputs[f"{node_id}.{slot}"] = (node_id, slot)
            for slot in cls._class_output_types():
                if (node_id, slot) not in connected_outputs:
                    ext_outputs[f"{node_id}.{slot}"] = (node_id, slot)
        return ext_inputs, ext_outputs

    def get_input_types(self) -> dict[str, type]:
        classes = PrimitiveComponent.registered_subclasses()
        result: dict[str, type] = {}
        for ext_name, (node_id, slot) in self._ext_inputs.items():
            node = self._sub_graph.nodes[node_id]
            cls = classes.get(node.type)
            if cls is not None:
                slot_types = cls._class_input_types()
                if slot in slot_types:
                    result[ext_name] = slot_types[slot]
        return result

    def get_output_types(self) -> dict[str, type]:
        classes = PrimitiveComponent.registered_subclasses()
        result: dict[str, type] = {}
        for ext_name, (node_id, slot) in self._ext_outputs.items():
            node = self._sub_graph.nodes[node_id]
            cls = classes.get(node.type)
            if cls is not None:
                slot_types = cls._class_output_types()
                if slot in slot_types:
                    result[ext_name] = slot_types[slot]
        return result

    def get_ui_input_types(self) -> dict[str, type]:
        return {}

    def get_ui_output_types(self) -> dict[str, type]:
        return {}

    def start(self, inputs: Any, outputs: Any) -> None:
        if self.status == Status.RUNNING:
            return
        self._status = Status.SETUP

        from src.core.graph import GraphManager

        self._inner_manager = GraphManager(self._sub_graph)

        for ext_name, (node_id, slot) in self._ext_inputs.items():
            outer_receiver = inputs.get(ext_name) if isinstance(inputs, dict) else None
            if outer_receiver is not None:
                self._inner_manager._receiver_handles[(node_id, slot)] = outer_receiver

        for ext_name, (node_id, slot) in self._ext_outputs.items():
            outer_sender = outputs.get(ext_name) if isinstance(outputs, dict) else None
            if outer_sender is not None:
                self._inner_manager._sender_handles[(node_id, slot)] = outer_sender

        self._inner_manager.run()
        self._status = Status.RUNNING

    def stop(self) -> None:
        super().stop()
        if self._inner_manager is not None:
            self._inner_manager.stop()
