from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.core.frames import (
    AudioFrame,
    DepthFrame,
    EOS,
    InterruptFrame,
    ObjectDetectionFrame,
    ObjectLocationFrame,
    ObjectSegmentationFrame,
    TextFrame,
    VideoDataFormat,
    VideoFrame,
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


def test_fish_tts_and_visualizers(monkeypatch) -> None:
    import src.core.conduit.tts_fish as fish_mod
    import src.core.conduit.depth_estimation_visualizer as depth_mod
    import src.core.conduit.object_detection_visualizer as det_mod
    import src.core.conduit.object_locator_visualizer as loc_mod
    import src.core.conduit.object_segmentation_visualizer as seg_mod

    class _FishClient:
        def __init__(self, api_key, base_url):
            self.api_key = api_key
            self.base_url = base_url
            self.tts = SimpleNamespace(stream_websocket=self.stream_websocket)
            self.stream_inputs = []

        def stream_websocket(self, text_iter, **kwargs):
            self.stream_inputs.append((list(text_iter), kwargs))
            return iter([b"\x01", b"\x02\x03", b"\x04"])

    monkeypatch.setattr(fish_mod, "FishAudio", _FishClient)
    fish = fish_mod.FishTTS(fish_mod.FishTTSConfig(api_key="k", base_url="u"))
    fish_audio = []
    fish.run(
        fish_mod.FishTTSInputs(
            text=_FakeRecv([TextFrame.new(text="hi"), EOS.END, None])
        ),
        fish_mod.FishTTSOutputs(
            audio=SimpleNamespace(send=lambda value: fish_audio.append(value))
        ),
    )
    assert len(fish_audio) == 2
    assert isinstance(fish_audio[0], AudioFrame)

    class _InterruptThread:
        def __init__(self, target, args=(), daemon=False):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(fish_mod.threading, "Thread", _InterruptThread)
    interrupted = fish_mod.FishTTS(fish_mod.FishTTSConfig(api_key="k", base_url="u"))
    interrupted_audio = []
    interrupted.run(
        fish_mod.FishTTSInputs(
            text=_FakeRecv([TextFrame.new(text="later"), None]),
            interrupt=_FakeRecv([InterruptFrame.new(reason="stop"), None]),
        ),
        fish_mod.FishTTSOutputs(
            audio=SimpleNamespace(send=lambda value: interrupted_audio.append(value))
        ),
    )
    assert interrupted_audio == []

    fake_cv2 = SimpleNamespace(
        COLORMAP_TURBO=1,
        INTER_NEAREST=2,
        FONT_HERSHEY_SIMPLEX=0,
        LINE_AA=16,
        applyColorMap=lambda data, cmap: np.stack([data, data, data], axis=-1),
        resize=lambda img, size, interpolation=None: np.zeros(
            (size[1], size[0], *(img.shape[2:] if img.ndim == 3 else ())),
            dtype=img.dtype if hasattr(img, "dtype") else np.uint8,
        ),
        addWeighted=lambda a, wa, b, wb, gamma: a,
        rectangle=lambda *args, **kwargs: None,
        getTextSize=lambda text, font, scale, thickness: ((len(text), 10), 0),
        putText=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(depth_mod, "cv2", fake_cv2)
    monkeypatch.setattr(det_mod, "cv2", fake_cv2)
    monkeypatch.setattr(loc_mod, "cv2", fake_cv2)
    monkeypatch.setattr(seg_mod, "cv2", fake_cv2)

    depth = np.array([[1.0, np.inf], [2.0, 0.0]], dtype=np.float32)
    colored = depth_mod.DepthEstimationVisualizer._colorize_depth(depth)
    assert colored.shape == (2, 2, 3)
    invalid_colored = depth_mod.DepthEstimationVisualizer._colorize_depth(
        np.array([[0.0, np.inf]], dtype=np.float32)
    )
    assert invalid_colored.shape == (1, 2, 3)
    overlay = depth_mod.DepthEstimationVisualizer._overlay(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint8),
    )
    assert overlay.shape == (4, 4, 3)

    depth_outputs = []
    depth_mod.DepthEstimationVisualizer().run(
        depth_mod.DepthEstimationVisualizerInputs(
            depth=_FakeRecv(
                [DepthFrame.new(data=depth), DepthFrame.new(data=depth), None]
            ),
            video=_FakeRecv(
                [
                    VideoFrame.new(
                        data=np.zeros((2, 2, 3), dtype=np.uint8),
                        format=VideoDataFormat.BGR,
                    )
                ]
            ),
        ),
        depth_mod.DepthEstimationVisualizerOutputs(
            video=SimpleNamespace(send=lambda value: depth_outputs.append(value))
        ),
    )
    assert len(depth_outputs) == 1
    depth_mod.DepthEstimationVisualizer().run(
        depth_mod.DepthEstimationVisualizerInputs(
            depth=_FakeRecv([None]),
            video=_FakeRecv([]),
        ),
        depth_mod.DepthEstimationVisualizerOutputs(
            video=SimpleNamespace(send=lambda value: depth_outputs.append(value))
        ),
    )

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    det_mod._draw_labeled_box(img, 1, 1, 5, 5, "obj", (0, 255, 0))
    det_frame = ObjectDetectionFrame.new(
        boxes=np.array([[[1, 1, 5, 5], [0, 0, 1, 1]]], dtype=np.float32),
        scores=np.array([[0.9, 0.1]], dtype=np.float32),
        prompts=("obj",),
    )
    det_mod._draw_detections(img, det_frame)
    det_outputs = []
    det_mod.ObjectDetectionVisualizer().run(
        det_mod.ObjectDetectionVisualizerInputs(
            detections=_FakeRecv([det_frame, det_frame, None]),
            video=_FakeRecv([VideoFrame.new(data=img, format=VideoDataFormat.BGR)]),
        ),
        det_mod.ObjectDetectionVisualizerOutputs(
            video=SimpleNamespace(send=lambda value: det_outputs.append(value))
        ),
    )
    assert len(det_outputs) == 1
    det_mod.ObjectDetectionVisualizer().run(
        det_mod.ObjectDetectionVisualizerInputs(
            detections=_FakeRecv([None]),
            video=_FakeRecv([]),
        ),
        det_mod.ObjectDetectionVisualizerOutputs(
            video=SimpleNamespace(send=lambda value: det_outputs.append(value))
        ),
    )

    loc_img = np.zeros((8, 8, 3), dtype=np.uint8)
    loc_frame = ObjectLocationFrame.new(
        labels=("obj", "skip-score", "skip-pos"),
        positions=np.array(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [np.nan, 0.0, 0.0]], dtype=np.float32
        ),
        depths=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        scores=np.array([0.9, 0.1, 0.9], dtype=np.float32),
        boxes=np.array([[0, 0, 5, 5], [0, 0, 2, 2], [1, 1, 3, 3]], dtype=np.float32),
        object_ids=np.array([1, 2, 3], dtype=np.int64),
    )
    loc_mod._draw_locations(loc_img, loc_frame)
    loc_outputs = []
    loc_mod.ObjectLocatorVisualizer().run(
        loc_mod.ObjectLocatorVisualizerInputs(
            locations=_FakeRecv([loc_frame, loc_frame, None]),
            video=_FakeRecv([VideoFrame.new(data=loc_img, format=VideoDataFormat.BGR)]),
        ),
        loc_mod.ObjectLocatorVisualizerOutputs(
            video=SimpleNamespace(send=lambda value: loc_outputs.append(value))
        ),
    )
    assert len(loc_outputs) == 1
    loc_mod.ObjectLocatorVisualizer().run(
        loc_mod.ObjectLocatorVisualizerInputs(
            locations=_FakeRecv([None]),
            video=_FakeRecv([]),
        ),
        loc_mod.ObjectLocatorVisualizerOutputs(
            video=SimpleNamespace(send=lambda value: loc_outputs.append(value))
        ),
    )

    seg_img = np.zeros((6, 6, 3), dtype=np.uint8)
    seg_frame = ObjectSegmentationFrame.new(
        masks=np.array(
            [
                np.array([[True, False], [False, True]], dtype=bool),
                np.array([[False, False], [False, False]], dtype=bool),
            ]
        ),
        boxes=np.array([[0, 0, 4, 4], [1, 1, 2, 2]], dtype=np.float32),
        scores=np.array([0.9, 0.1], dtype=np.float32),
        object_ids=np.array([7, 8], dtype=np.int64),
        labels=("cat", "dog"),
    )
    seg_mod._draw_segmentations(seg_img, seg_frame, score_threshold=0.5)
    seg_outputs = []
    seg_mod.ObjectSegmentationVisualizer(
        seg_mod.ObjectSegmentationVisualizerConfig(score_threshold=0.5)
    ).run(
        seg_mod.ObjectSegmentationVisualizerInputs(
            segmentations=_FakeRecv([seg_frame, seg_frame, None]),
            video=_FakeRecv([VideoFrame.new(data=seg_img, format=VideoDataFormat.BGR)]),
        ),
        seg_mod.ObjectSegmentationVisualizerOutputs(
            video=SimpleNamespace(send=lambda value: seg_outputs.append(value))
        ),
    )
    assert len(seg_outputs) == 1
    seg_mod.ObjectSegmentationVisualizer(
        seg_mod.ObjectSegmentationVisualizerConfig(score_threshold=0.5)
    ).run(
        seg_mod.ObjectSegmentationVisualizerInputs(
            segmentations=_FakeRecv([None]),
            video=_FakeRecv([]),
        ),
        seg_mod.ObjectSegmentationVisualizerOutputs(
            video=SimpleNamespace(send=lambda value: seg_outputs.append(value))
        ),
    )
