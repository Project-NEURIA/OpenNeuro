"""RemoteDepthEstimator conduit — FFS + TSDF depth via a remote GPU server."""

from __future__ import annotations

import io
import json
from typing import Literal, NamedTuple

import cv2
import numpy as np
import requests
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import Tag, ThreadedComponent
from src.core.frames import (
    DepthFrame,
    StereoCameraParamsFrame,
    StereoVideoFrame,
    VideoDataFormat,
)
from src.core.utils import resize_and_crop


def _to_nvblox_c2w(extrinsics: np.ndarray) -> list[float]:
    """Convert pipeline extrinsics to OpenCV camera-to-world (row-major list).

    Pipeline extrinsics are world-to-camera with VRChat's Z-flip applied.
    The server expects camera-to-world in OpenCV convention.
    """
    # Undo VRChat's Z-flip to recover raw OpenVR world-to-camera.
    w2c = extrinsics.copy().astype(np.float32)
    w2c[2, :] *= -1
    w2c[:, 2] *= -1
    # Invert to camera-to-world.
    c2w = np.linalg.inv(w2c)
    # OpenGL -> OpenCV: flip Y and Z columns.
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    return c2w.flatten().tolist()


class RemoteDepthEstimatorConfig(BaseModel):
    server_url: str = "http://localhost:8000"
    resize_width: int = 512
    resize_height: int = 512
    jpeg_quality: int = 85
    timeout: float = 10.0
    valid_iters: int = 4
    max_disp: int = 192
    render_max_steps: int = 100
    decay_every: int = 100
    voxel_size: float = 0.05
    max_integ_dist: float = 10.0
    trunc_dist_vox: float = 3.0
    max_weight: float = 3.0
    weighting_mode: Literal["constant", "inverse_distance"] = "constant"
    invalid_depth_decay: float = 0.9
    decay_factor: float = 0.95
    decay_threshold: float = 0.001


class RemoteDepthEstimatorInputs(NamedTuple):
    stereo_video: Receiver[StereoVideoFrame]
    camera_params: Receiver[StereoCameraParamsFrame]


class RemoteDepthEstimatorOutputs(NamedTuple):
    live_depth: Sender[DepthFrame]
    tsdf_depth: Sender[DepthFrame]
    stereo_video: Sender[StereoVideoFrame]


class RemoteDepthEstimator(
    ThreadedComponent[RemoteDepthEstimatorInputs, RemoteDepthEstimatorOutputs]
):
    """Offloads FFS stereo depth + TSDF fusion to a remote GPU server.

    Replaces StereoDepthEstimator + TSDFMapper in the graph when using a
    cloud GPU.  Sends JPEG-compressed stereo frames to the server and
    receives back both live (per-frame FFS) and TSDF-rendered stable
    depth maps.
    """

    description = "Remote FFS + TSDF depth estimation via GPU server"
    tags = Tag(io={"conduit"}, functionality={"image"})

    def __init__(self, config: RemoteDepthEstimatorConfig) -> None:
        super().__init__()
        self.config = config
        self._session_id: str | None = None
        self._frame_count = 0

    def _create_session(self, cam: StereoCameraParamsFrame) -> None:
        """Create a server session with scaled intrinsics matching resize dims."""
        cfg = self.config
        K = cam.intrinsics.astype(np.float64)
        scale_x = cfg.resize_width / cam.width
        scale_y = cfg.resize_height / cam.height

        resp = requests.post(
            f"{cfg.server_url}/session",
            json={
                "fx": float(K[0, 0] * scale_x),
                "fy": float(K[1, 1] * scale_y),
                "cx": float(K[0, 2] * scale_x),
                "cy": float(K[1, 2] * scale_y),
                "baseline": float(cam.baseline),
                "valid_iters": cfg.valid_iters,
                "max_disp": cfg.max_disp,
                "voxel_size": cfg.voxel_size,
                "max_integ_dist": cfg.max_integ_dist,
                "trunc_dist_vox": cfg.trunc_dist_vox,
                "max_weight": cfg.max_weight,
                "weighting_mode": cfg.weighting_mode,
                "invalid_depth_decay": cfg.invalid_depth_decay,
                "decay_factor": cfg.decay_factor,
                "decay_threshold": cfg.decay_threshold,
                "render_max_steps": cfg.render_max_steps,
            },
            timeout=cfg.timeout,
        )
        resp.raise_for_status()
        self._session_id = resp.json()["session_id"]
        print(
            f"[RemoteDepthEstimator] Session {self._session_id} created on {cfg.server_url}"
        )

    def _delete_session(self) -> None:
        if self._session_id is None:
            return
        try:
            requests.delete(
                f"{self.config.server_url}/session/{self._session_id}",
                timeout=self.config.timeout,
            )
        except Exception as e:
            print(f"[RemoteDepthEstimator] Failed to delete session: {e}")
        self._session_id = None

    def _encode_jpeg(self, img_bgr: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(
            ".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buf.tobytes()

    def stop(self) -> None:
        super().stop()
        self._delete_session()

    def run(
        self,
        inputs: RemoteDepthEstimatorInputs,
        outputs: RemoteDepthEstimatorOutputs,
    ) -> None:
        cfg = self.config
        inputs.stereo_video.newest = True
        inputs.camera_params.newest = True

        for stereo_frame in inputs.stereo_video:
            if stereo_frame is None:
                break

            cam_frame = next(inputs.camera_params, None)
            if cam_frame is None:
                break

            # Create session on first frame.
            if self._session_id is None:
                try:
                    self._create_session(cam_frame)
                except Exception as e:
                    print(f"[RemoteDepthEstimator] Session creation failed: {e}")
                    continue

            # Resize stereo images.
            rw, rh = cfg.resize_width, cfg.resize_height
            left_bgr = resize_and_crop(
                stereo_frame.get("left", VideoDataFormat.BGR), rw, rh
            )
            right_bgr = resize_and_crop(
                stereo_frame.get("right", VideoDataFormat.BGR), rw, rh
            )

            # Convert extrinsics to OpenCV c2w for the server.
            pose_list = _to_nvblox_c2w(cam_frame.extrinsics)

            # Send to server.
            try:
                resp = requests.post(
                    f"{cfg.server_url}/session/{self._session_id}/frame",
                    files={
                        "left_image": (
                            "left.jpg",
                            self._encode_jpeg(left_bgr),
                            "image/jpeg",
                        ),
                        "right_image": (
                            "right.jpg",
                            self._encode_jpeg(right_bgr),
                            "image/jpeg",
                        ),
                    },
                    data={
                        "pose": json.dumps(pose_list),
                    },
                    timeout=cfg.timeout,
                )
                resp.raise_for_status()
            except Exception as e:
                print(f"[RemoteDepthEstimator] Frame request failed: {e}")
                continue

            # Decode response.
            npz = np.load(io.BytesIO(resp.content))
            d_live = npz["d_live"].astype(np.float32)
            d_map = npz["d_map"].astype(np.float32)

            outputs.live_depth.send(DepthFrame.new(data=d_live, is_metric=True))
            outputs.tsdf_depth.send(DepthFrame.new(data=d_map, is_metric=True))
            outputs.stereo_video.send(
                StereoVideoFrame.new(
                    left=left_bgr,
                    right=right_bgr,
                    format=VideoDataFormat.BGR,
                )
            )

            self._frame_count += 1

            # Periodic TSDF decay.
            if cfg.decay_every > 0 and self._frame_count % cfg.decay_every == 0:
                try:
                    requests.post(
                        f"{cfg.server_url}/session/{self._session_id}/decay",
                        timeout=cfg.timeout,
                    )
                except Exception as e:
                    print(f"[RemoteDepthEstimator] Decay request failed: {e}")

        self._delete_session()
        print("[RemoteDepthEstimator] Stopped")
