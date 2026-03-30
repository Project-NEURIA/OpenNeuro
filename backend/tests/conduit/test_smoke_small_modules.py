from __future__ import annotations

import importlib
import sys
import types

import numpy as np

from src.core.conduit.buffer import Buffer, BufferInputs, BufferOutputs
from src.core.conduit.do_nothing_tool import (
    DoNothingTool,
    DoNothingToolInputs,
    DoNothingToolOutputs,
)
from src.core.conduit.messages_to_text import (
    MessagesToText,
    MessagesToTextInputs,
    MessagesToTextOutputs,
)
from src.core.conduit.movement_tool import (
    MovementTool,
    MovementToolInputs,
    MovementToolOutputs,
)
from src.core.conduit.passthrough import (
    Passthrough,
    PassthroughInputs,
    PassthroughOutputs,
)
from src.core.conduit.stereo_camera_params_adapter import (
    StereoCameraParamsAdapter,
    StereoCameraParamsAdapterInputs,
    StereoCameraParamsAdapterOutputs,
)
from src.core.frames import (
    CameraParamsFrame,
    EOS,
    MessageFrame,
    StereoCameraParamsFrame,
    ToolCall,
)


def _ensure_transformers_patch(monkeypatch) -> None:
    import transformers.utils.generic as transformers_generic

    monkeypatch.setattr(
        transformers_generic,
        "check_model_inputs",
        lambda *args, **kwargs: lambda func: func,
        raising=False,
    )


class _FakeRecv:
    def __init__(self, items):
        self._items = list(items)
        self._iter = iter(self._items)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def __call__(self, *args, **kwargs):
        return iter(self._items)


def test_small_conduit_modules_run_paths() -> None:
    batches = []
    buf = Buffer[int]()
    buf.run(
        BufferInputs(data=_FakeRecv([1, 2, EOS.END, EOS.END, 3, None])),
        BufferOutputs(batch=types.SimpleNamespace(send=lambda x: batches.append(x))),
    )
    assert batches == [[1, 2]]

    texts = []
    msg_to_text = MessagesToText()
    msg_to_text.run(
        MessagesToTextInputs(
            messages=_FakeRecv(
                [
                    [
                        MessageFrame.new(role="user", content="hi"),
                        MessageFrame.new(role="assistant", content="hello"),
                    ],
                    None,
                ]
            )
        ),
        MessagesToTextOutputs(
            text=types.SimpleNamespace(send=lambda x: texts.append(x))
        ),
    )
    assert texts[0].get() == "user: hi\nassistant: hello"

    passed = []
    passthrough = Passthrough[int]()
    passthrough.run(
        PassthroughInputs(data=_FakeRecv([1, 2, None])),
        PassthroughOutputs(data=types.SimpleNamespace(send=lambda x: passed.append(x))),
    )
    assert passed == [1, 2]

    tool_defs = []
    tool_results = []
    do_nothing = DoNothingTool()
    do_nothing.setup(
        DoNothingToolOutputs(
            tool_def=types.SimpleNamespace(send=lambda x: tool_defs.append(x)),
            tool_result=types.SimpleNamespace(send=lambda x: tool_results.append(x)),
        )
    )
    do_nothing.run(
        DoNothingToolInputs(
            tool_call=_FakeRecv(
                [ToolCall.new(call_id="1", name="x", arguments="{}"), None]
            )
        ),
        DoNothingToolOutputs(
            tool_def=types.SimpleNamespace(send=lambda x: tool_defs.append(x)),
            tool_result=types.SimpleNamespace(send=lambda x: tool_results.append(x)),
        ),
    )
    assert tool_defs[0].name == "do_nothing"
    assert tool_results[0].call_id == "1"

    move_defs = []
    move_results = []
    goals = []
    instructions = []
    movement = MovementTool()
    movement.setup(
        MovementToolOutputs(
            tool_def=types.SimpleNamespace(send=lambda x: move_defs.append(x)),
            tool_result=types.SimpleNamespace(send=lambda x: move_results.append(x)),
            goal=types.SimpleNamespace(send=lambda x: goals.append(x)),
            instruction=types.SimpleNamespace(send=lambda x: instructions.append(x)),
        )
    )
    movement.run(
        MovementToolInputs(
            tool_call=_FakeRecv(
                [
                    ToolCall.new(call_id="a", name="other", arguments="{}"),
                    ToolCall.new(
                        call_id="b",
                        name="move",
                        arguments='{"instruction":"walk","x":1,"z":2}',
                    ),
                    ToolCall.new(
                        call_id="c",
                        name="move",
                        arguments='{"instruction":"dance"}',
                    ),
                    ToolCall.new(
                        call_id="d",
                        name="move",
                        arguments='{"instruction":"turn","heading":45}',
                    ),
                    None,
                ]
            )
        ),
        MovementToolOutputs(
            tool_def=types.SimpleNamespace(send=lambda x: move_defs.append(x)),
            tool_result=types.SimpleNamespace(send=lambda x: move_results.append(x)),
            goal=types.SimpleNamespace(send=lambda x: goals.append(x)),
            instruction=types.SimpleNamespace(send=lambda x: instructions.append(x)),
        ),
    )
    assert move_defs[0].name == "move"
    assert move_results[0].content == "Executing 'walk' toward (1, 2)"
    assert move_results[1].content == "Executing 'dance'"
    assert move_results[2].content == "Executing 'turn' facing 45°"
    assert goals[0].x == 1.0 and goals[0].z == 2.0
    assert goals[1].heading == 45.0
    assert [t.get() for t in instructions] == ["walk", "dance", "turn"]

    adapted = []
    adapter = StereoCameraParamsAdapter()
    stereo = StereoCameraParamsFrame.new(
        intrinsics=np.eye(3, dtype=np.float32),
        extrinsics=np.eye(4, dtype=np.float32),
        baseline=0.1,
        width=2,
        height=3,
    )
    adapter.run(
        StereoCameraParamsAdapterInputs(stereo_camera_params=_FakeRecv([stereo, None])),
        StereoCameraParamsAdapterOutputs(
            camera_params=types.SimpleNamespace(send=lambda x: adapted.append(x))
        ),
    )
    assert isinstance(adapted[0], CameraParamsFrame)
    assert adapted[0].width == 2 and adapted[0].height == 3


def test_heavy_module_import_smoke(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(
        sys.modules,
        "ovd_client",
        types.SimpleNamespace(
            Client=type("Client", (), {}),
            Pose=type(
                "Pose",
                (),
                {"__init__": lambda self, **kwargs: self.__dict__.update(kwargs)},
            ),
        ),
    )
    _ensure_transformers_patch(monkeypatch)

    qwen_voice_dir = tmp_path / "voices"
    qwen_voice_dir.mkdir()
    (qwen_voice_dir / "a.wav").write_bytes(b"fake")
    (qwen_voice_dir / "a.txt").write_text("hello", encoding="utf-8")

    module_names = [
        "src.core.conduit.qwen_tts.tts_model.configuration_qwen3_tts",
        "src.core.conduit.qwen_tts.tts_model.configuration_qwen3_tts_tokenizer_v2",
        "src.core.conduit.qwen_tts.tts_model.modeling_qwen3_tts",
        "src.core.conduit.qwen_tts.tts_model.modeling_qwen3_tts_tokenizer_v2",
        "src.core.conduit.qwen_tts.tts_model.processing_qwen3_tts",
        "src.core.conduit.qwen_tts.tts_model.qwen3_tts_tokenizer",
        "src.core.conduit.qwen_tts.tts_model",
        "src.core.conduit.qwen_tts.model",
        "src.core.conduit.qwen_tts.component",
        "src.core.sink.openvr_movement",
        "src.core.sink.osc_chatbox",
        "src.core.sink.osc_face",
    ]

    imported = []
    for name in module_names:
        imported.append(importlib.reload(importlib.import_module(name)))

    qwen_component = imported[8]
    options = qwen_component.QwenTTS.get_options(
        {"ref_samples_dir": str(qwen_voice_dir)}
    )
    assert options["config"]["voice_id"][0]["value"] == "a"

    no_options = qwen_component.QwenTTS.get_options(
        {"ref_samples_dir": str(tmp_path / "missing")}
    )
    assert no_options == {}

    assert imported[9]._to_ovd_pose(None) is None
