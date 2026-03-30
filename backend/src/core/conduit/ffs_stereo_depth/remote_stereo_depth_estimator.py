"""RemoteStereoDepthEstimator — offloads FFS depth + TSDF fusion to a remote WebSocket server."""

from __future__ import annotations

import logging
import threading
import time
from typing import NamedTuple

import cv2
import numpy as np
from pydantic import BaseModel

from websockets.sync.client import connect, ClientConnection

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.conduit.ffs_stereo_depth.depth_ws_protocol import (
    pack_init,
    unpack_init_ack,
    pack_frame,
    unpack_frame_result,
    pack_decay,
    pack_clear,
    pack_close,
    msg_tag,
    unpack_error,
    MSG_FRAME_RESULT,
    MSG_DECAY_ACK,
    MSG_CLEAR_ACK,
    MSG_ERROR,
    MSG_INIT_ACK,
)
from src.core.frames import (
    DepthFrame,
    StereoCameraParamsFrame,
    StereoVideoFrame,
    VideoDataFormat,
)

logger = logging.getLogger(__name__)


class RemoteStereoDepthEstimatorConfig(BaseModel):
    server_url: str = "ws://localhost:8000/ws"
    jpeg_quality: int = 90
    resize_width: int = 512
    resize_height: int = 512
    min_depth: float = 0.2
    max_depth: float = 10.0
    # TSDF config (sent to server in INIT)
    voxel_size: float = 0.05
    max_integ_dist: float = 10.0
    trunc_dist_vox: float = 3.0
    max_weight: float = 3.0
    weighting_mode: str = "constant"
    invalid_depth_decay: float = 0.9
    mesh_min_weight: float = 0.1
    no_mesh_weld: bool = False
    decay_factor: float = 0.95
    decay_threshold: float = 0.001
    render_max_steps: int = 100
    # Maintenance
    decay_every: int = 100
    # Reconnection
    max_reconnect_delay: float = 10.0


class RemoteStereoDepthEstimatorInputs(NamedTuple):
    stereo_video: Receiver[StereoVideoFrame]
    camera_params: Receiver[StereoCameraParamsFrame]


class RemoteStereoDepthEstimatorOutputs(NamedTuple):
    depth: Sender[DepthFrame]
    stereo_video: Sender[StereoVideoFrame]


class RemoteStereoDepthEstimator(
    ThreadedComponent[
        RemoteStereoDepthEstimatorInputs,
        RemoteStereoDepthEstimatorOutputs,
    ]
):
    """Stereo depth + TSDF fusion via a remote WebSocket server.

    Drop-in replacement for StereoDepthEstimator.  Instead of running
    FFS locally, it streams stereo frames to a remote GPU server that
    performs FFS depth estimation + nvblox TSDF fusion and returns the
    fused depth map.
    """

    description = "Estimates depth from stereo video via remote FFS+TSDF server"
    tags = Tag(io={"conduit"}, functionality={"image"}, gpu={"cpu"})

    def __init__(self, config: RemoteStereoDepthEstimatorConfig) -> None:
        super().__init__()
        self.config = config
        self._ws: ClientConnection | None = None

    def stop(self) -> None:
        super().stop()
        self._close_ws()

    def _close_ws(self) -> None:
        ws = self._ws
        if ws is not None:
            try:
                ws.send(pack_close())
            except Exception:
                pass
            try:
                ws.close()
            except Exception:
                pass
            self._ws = None

    @staticmethod
    def _resize_for_inference(img: np.ndarray, width: int, height: int) -> np.ndarray:
        h_src, w_src = img.shape[:2]
        scale = max(width / w_src, height / h_src)
        w_s = int(round(w_src * scale))
        h_s = int(round(h_src * scale))
        resized = cv2.resize(img, (w_s, h_s), interpolation=cv2.INTER_AREA)
        y0 = (h_s - height) // 2
        x0 = (w_s - width) // 2
        return resized[y0 : y0 + height, x0 : x0 + width]

    def _build_init_config(self, cam_frame: StereoCameraParamsFrame) -> dict:
        """Build the INIT message config from camera params + component config."""
        K = cam_frame.intrinsics.astype(np.float32)
        # Scale intrinsics to the inference resolution
        sx = self.config.resize_width / cam_frame.width
        sy = self.config.resize_height / cam_frame.height
        fx = float(K[0, 0] * sx)
        fy = float(K[1, 1] * sy)
        cx = float(K[0, 2] * sx)
        cy = float(K[1, 2] * sy)

        c = self.config
        return {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "baseline": float(cam_frame.baseline),
            "width": c.resize_width,
            "height": c.resize_height,
            "min_depth": c.min_depth,
            "max_depth": c.max_depth,
            "voxel_size": c.voxel_size,
            "max_integ_dist": c.max_integ_dist,
            "trunc_dist_vox": c.trunc_dist_vox,
            "max_weight": c.max_weight,
            "weighting_mode": c.weighting_mode,
            "invalid_depth_decay": c.invalid_depth_decay,
            "mesh_min_weight": c.mesh_min_weight,
            "no_mesh_weld": c.no_mesh_weld,
            "decay_factor": c.decay_factor,
            "decay_threshold": c.decay_threshold,
            "render_max_steps": c.render_max_steps,
            "jpeg_quality": c.jpeg_quality,
        }

    def _connect_and_init(
        self, cam_frame: StereoCameraParamsFrame
    ) -> ClientConnection:
        """Open WebSocket, send INIT, wait for INIT_ACK."""
        ws = connect(self.config.server_url,
                     max_size=16 * 1024 * 1024,
                     ping_timeout=120,
                     ping_interval=30)
        self._ws = ws
        ws.send(pack_init(self._build_init_config(cam_frame)))
        ack = ws.recv()
        tag = msg_tag(ack)
        if tag == MSG_ERROR:
            raise RuntimeError(f"Server error on INIT: {unpack_error(ack)}")
        if tag != MSG_INIT_ACK:
            raise RuntimeError(f"Unexpected response to INIT: 0x{tag:02x}")
        info = unpack_init_ack(ack)
        logger.info("Connected to %s — VRAM %.0f/%.0f MB",
                     self.config.server_url,
                     info.get("vram_used_mb", 0),
                     info.get("vram_total_mb", 0))
        return ws

    def run(
        self,
        inputs: RemoteStereoDepthEstimatorInputs,
        outputs: RemoteStereoDepthEstimatorOutputs,
    ) -> None:
        inputs.stereo_video.newest = True
        inputs.camera_params.newest = True

        print("[RemoteStereoDepthEstimator] Waiting for camera params …")
        cam_frame = next(inputs.camera_params, None)
        if cam_frame is None:
            print("[RemoteStereoDepthEstimator] No camera params — stopping")
            return

        ws: ClientConnection | None = None
        reconnect_delay = 1.0
        frame_count = 0

        print(f"[RemoteStereoDepthEstimator] Connecting to {self.config.server_url}")

        for stereo_frame in inputs.stereo_video:
            if stereo_frame is None or self.stop_event.is_set():
                break

            # Update camera params if a newer one is available
            new_cam = next(inputs.camera_params, None)
            if new_cam is not None:
                cam_frame = new_cam

            # Ensure connection
            if ws is None:
                try:
                    ws = self._connect_and_init(cam_frame)
                    reconnect_delay = 1.0
                except Exception as exc:
                    logger.warning("Connection failed: %s (retry in %.0fs)",
                                   exc, reconnect_delay)
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(
                        reconnect_delay * 2, self.config.max_reconnect_delay
                    )
                    continue

            # Resize to inference resolution
            rw, rh = self.config.resize_width, self.config.resize_height
            left_rgb = self._resize_for_inference(
                stereo_frame.get("left", VideoDataFormat.RGB), rw, rh
            )
            right_rgb = self._resize_for_inference(
                stereo_frame.get("right", VideoDataFormat.RGB), rw, rh
            )

            # JPEG encode
            q = self.config.jpeg_quality
            _, left_jpg = cv2.imencode(
                ".jpg",
                cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, q],
            )
            _, right_jpg = cv2.imencode(
                ".jpg",
                cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, q],
            )

            # Build pose (camera-to-world).
            # StereoCameraParamsFrame.extrinsics is world-to-camera.
            w2c = cam_frame.extrinsics.astype(np.float32)
            c2w = np.linalg.inv(w2c)

            try:
                ws.send(pack_frame(bytes(left_jpg), bytes(right_jpg), c2w))
                resp = ws.recv()
            except Exception as exc:
                logger.warning("WebSocket error: %s — reconnecting", exc)
                self._close_ws()
                ws = None
                continue

            tag = msg_tag(resp)
            if tag == MSG_ERROR:
                logger.warning("Server error: %s", unpack_error(resp))
                continue
            if tag != MSG_FRAME_RESULT:
                logger.warning("Unexpected message 0x%02x", tag)
                continue

            result = unpack_frame_result(resp)
            frame_count += 1

            if result.vram_warning:
                logger.warning("Server VRAM warning — sending CLEAR")
                try:
                    ws.send(pack_clear())
                    ws.recv()  # CLEAR_ACK
                except Exception:
                    pass

            # Emit depth
            if result.d_map is not None:
                outputs.depth.send(
                    DepthFrame.new(data=result.d_map, is_metric=True)
                )

            # Emit resized stereo (decode server JPEGs or use local resized)
            if result.left_jpeg is not None and result.right_jpeg is not None:
                left_out = cv2.imdecode(
                    np.frombuffer(result.left_jpeg, np.uint8),
                    cv2.IMREAD_COLOR,
                )
                right_out = cv2.imdecode(
                    np.frombuffer(result.right_jpeg, np.uint8),
                    cv2.IMREAD_COLOR,
                )
            else:
                left_out = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
                right_out = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)

            outputs.stereo_video.send(
                StereoVideoFrame.new(
                    left=left_out,
                    right=right_out,
                    format=VideoDataFormat.BGR,
                )
            )

            # Periodic TSDF decay
            if (
                self.config.decay_every > 0
                and frame_count % self.config.decay_every == 0
                and ws is not None
            ):
                try:
                    ws.send(pack_decay())
                    ws.recv()  # DECAY_ACK
                except Exception:
                    pass

        # Cleanup
        self._close_ws()
        print(f"[RemoteStereoDepthEstimator] Stopped ({frame_count} frames)")
