"""TSDFMapper conduit — TSDF volume fusion via nvblox with stable depth rendering."""

from __future__ import annotations

import threading
import time
from typing import Any, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import Tag, ThreadedComponent
from src.core.frames import CameraParamsFrame, DepthFrame, VideoDataFormat, VideoFrame
from src.core.utils import auto_device

try:
    from nvblox_torch.mapper import Mapper
    from nvblox_torch.mapper_params import MapperParams
    from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
    from nvblox_torch.rendering import render_depth_image
    from nvblox_torch.sensor import Sensor

    _NVBLOX_AVAILABLE = True
except (ImportError, OSError):
    _NVBLOX_AVAILABLE = False

try:
    import open3d as _o3d  # noqa: F401

    _OPEN3D_AVAILABLE = True
except ImportError:
    _OPEN3D_AVAILABLE = False

# nvblox weighting modes (must match nvblox_torch enum).
_WEIGHTING_MODES: dict[str, int] = {
    "constant": 0,
    "inverse_distance": 1,
}


class TSDFMapperConfig(BaseModel):
    voxel_size: float = 0.05
    max_integration_distance: float = 10.0
    truncation_distance_vox: float = 3.0
    max_weight: float = 3.0
    weighting_mode: Literal["constant", "inverse_distance"] = "constant"
    invalid_depth_decay: float = 0.9
    decay_every: int = 100
    decay_factor: float = 0.95
    decay_threshold: float = 0.001
    max_vram_mb: float = 7000
    render_max_steps: int = 100
    show_visualizer: bool = False
    mesh_update_every: int = 5
    mesh_min_weight: float = 0.1


class TSDFMapperInputs(NamedTuple):
    depth: Receiver[DepthFrame]
    video: Receiver[VideoFrame]
    camera_params: Receiver[CameraParamsFrame]


class TSDFMapperOutputs(NamedTuple):
    depth: Sender[DepthFrame]


class _MeshViewer:
    """Non-blocking Open3D mesh viewer running in a background daemon thread."""

    def __init__(self) -> None:
        import open3d as o3d

        self._o3d = o3d
        self._lock = threading.Lock()
        self._mesh: Any = None
        self._updated = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        o3d = self._o3d
        vis = o3d.visualization.Visualizer()
        try:
            vis.create_window("TSDF Mesh", width=1280, height=720)
        except Exception as e:
            print(f"[TSDFMapper] Could not create Open3D window: {e}")
            self._stop.set()
            return

        opt = vis.get_render_option()
        if opt is not None:
            opt.mesh_show_back_face = True

        current_mesh = o3d.geometry.TriangleMesh()
        vis.add_geometry(current_mesh)
        first_update = True

        while not self._stop.is_set():
            with self._lock:
                if self._updated and self._mesh is not None:
                    current_mesh.vertices = self._mesh.vertices
                    current_mesh.triangles = self._mesh.triangles
                    current_mesh.vertex_colors = self._mesh.vertex_colors
                    current_mesh.vertex_normals = self._mesh.vertex_normals
                    self._updated = False
                    need_update = True
                else:
                    need_update = False

            if need_update:
                vis.update_geometry(current_mesh)
                if first_update:
                    vis.reset_view_point(True)
                    first_update = False

            if not vis.poll_events():
                self._stop.set()
                break
            vis.update_renderer()
            time.sleep(0.016)  # ~60 Hz

        vis.destroy_window()

    def update(self, o3d_mesh: Any) -> None:
        with self._lock:
            self._mesh = o3d_mesh
            self._updated = True

    def close(self) -> None:
        self._stop.set()


def _to_nvblox_c2w(extrinsics: np.ndarray) -> Any:
    """Convert CameraParamsFrame extrinsics to nvblox camera-to-world (OpenCV convention).

    The pipeline extrinsics have VRChat's Z-flip applied (left-handed, +Z forward).
    nvblox expects camera-to-world in OpenCV convention (right-handed, +Z into screen).
    We undo VRChat's flip to recover the raw OpenVR w2c, invert, then flip Y and Z.
    """
    import torch

    # Undo VRChat's Z-flip to recover raw OpenVR world-to-camera.
    w2c = extrinsics.copy().astype(np.float32)
    w2c[2, :] *= -1
    w2c[:, 2] *= -1
    # Invert to camera-to-world.
    c2w = np.linalg.inv(w2c)
    # OpenGL -> OpenCV: flip Y and Z columns.
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    return torch.from_numpy(c2w).float()


class TSDFMapper(ThreadedComponent[TSDFMapperInputs, TSDFMapperOutputs]):
    """Integrates depth frames into a TSDF volume and renders stable depth via sphere-tracing.

    Uses nvblox for real-time TSDF fusion. Each incoming depth frame + colour image
    is integrated into the volume using the camera pose. A sphere-traced depth map
    is rendered from the current viewpoint and emitted downstream, providing
    temporally stable depth compared to raw per-frame stereo estimation.
    """

    description = "TSDF volume fusion with stable depth rendering via nvblox"
    tags = Tag(io={"conduit"}, functionality={"image"}, gpu={"nvidia"})

    def __init__(self, config: TSDFMapperConfig) -> None:
        super().__init__()
        self.config = config
        self._device = auto_device("auto")
        self._mapper: Any = None
        self._sensor: Any = None
        self._viewer: _MeshViewer | None = None
        self._frame_count = 0

    def setup(self, outputs: TSDFMapperOutputs) -> None:
        if not _NVBLOX_AVAILABLE:
            raise RuntimeError(
                "nvblox_torch is not installed. "
                "Build and install it from https://github.com/nvidia-isaac/nvblox: "
                "pip install -e libs/nvblox/nvblox_torch"
            )

        import torch

        torch.set_grad_enabled(False)

        cfg = self.config
        params = MapperParams()

        proj = params.get_projective_integrator_params()
        proj._c_params.set_projective_integrator_max_integration_distance_m(
            cfg.max_integration_distance
        )
        proj._c_params.set_projective_integrator_truncation_distance_vox(
            cfg.truncation_distance_vox
        )
        proj._c_params.set_projective_integrator_max_weight(cfg.max_weight)
        proj._c_params.set_projective_integrator_weighting_mode(
            _WEIGHTING_MODES[cfg.weighting_mode]
        )
        proj._c_params.set_projective_tsdf_integrator_invalid_depth_decay_factor(
            cfg.invalid_depth_decay
        )
        params.set_projective_integrator_params(proj)

        mesh_p = params.get_mesh_integrator_params()
        mesh_p._c_params.set_mesh_integrator_min_weight(cfg.mesh_min_weight)
        params.set_mesh_integrator_params(mesh_p)

        decay_p = params.get_tsdf_decay_integrator_params()
        decay_p._c_params.set_tsdf_decay_factor(cfg.decay_factor)
        decay_p._c_params.set_tsdf_decayed_weight_threshold(cfg.decay_threshold)
        params.set_tsdf_decay_integrator_params(decay_p)

        self._mapper = Mapper(
            voxel_sizes_m=cfg.voxel_size,
            integrator_types=ProjectiveIntegratorType.TSDF,
            mapper_parameters=params,
        )

        if cfg.show_visualizer:
            if not _OPEN3D_AVAILABLE:
                print("[TSDFMapper] open3d not installed, skipping 3D visualizer")
            else:
                self._viewer = _MeshViewer()

        print(
            f"[TSDFMapper] Ready  voxel_size={cfg.voxel_size}m  device={self._device}"
        )

    def _ensure_sensor(
        self, fx: float, fy: float, cx: float, cy: float, W: int, H: int
    ) -> None:
        if self._sensor is not None:
            return
        self._sensor = Sensor.from_camera(fu=fx, fv=fy, cu=cx, cv=cy, width=W, height=H)

    def _check_vram(self) -> None:
        import torch

        if self.config.max_vram_mb <= 0:
            return
        try:
            free, total = torch.cuda.mem_get_info()
            used_mb = (total - free) / (1024 * 1024)
        except Exception:
            return
        if used_mb > self.config.max_vram_mb:
            print(f"[TSDFMapper] VRAM {used_mb:.0f}MB > cap, clearing map")
            self._mapper.clear()
            self._sensor = None

    def stop(self) -> None:
        super().stop()
        if self._viewer is not None:
            self._viewer.close()

    def run(self, inputs: TSDFMapperInputs, outputs: TSDFMapperOutputs) -> None:
        import torch

        inputs.depth.newest = True
        inputs.video.newest = True
        inputs.camera_params.newest = True

        for depth_frame in inputs.depth:
            if depth_frame is None:
                break

            video_frame = next(inputs.video, None)
            if video_frame is None:
                break

            cam_frame = next(inputs.camera_params, None)
            if cam_frame is None:
                break

            depth_data = depth_frame.data  # (Hd, Wd) float32
            Hd, Wd = depth_data.shape[:2]

            # Scale intrinsics from source resolution to depth resolution.
            K = cam_frame.intrinsics.astype(np.float64)
            scale_x = Wd / cam_frame.width
            scale_y = Hd / cam_frame.height
            fx = float(K[0, 0] * scale_x)
            fy = float(K[1, 1] * scale_y)
            cx = float(K[0, 2] * scale_x)
            cy = float(K[1, 2] * scale_y)

            self._ensure_sensor(fx, fy, cx, cy, Wd, Hd)

            # Camera pose for nvblox (OpenCV camera-to-world).
            t_w_c = _to_nvblox_c2w(cam_frame.extrinsics)

            # Convert depth and colour to GPU tensors.
            depth_gpu = torch.as_tensor(depth_data, device=self._device)
            color_rgb = video_frame.get(VideoDataFormat.RGB)
            color_gpu = torch.as_tensor(color_rgb, device=self._device)

            # Integrate into TSDF.
            self._mapper.add_depth_frame(depth_gpu, t_w_c, self._sensor)
            self._mapper.add_color_frame(color_gpu, t_w_c, self._sensor)
            self._frame_count += 1

            # Sphere-trace the TSDF for stable depth.
            K_torch = torch.tensor(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32
            )
            rendered = render_depth_image(
                self._mapper.tsdf_layer_view(),
                t_w_c,
                K_torch,
                Hd,
                Wd,
                max_ray_length=float(self.config.max_integration_distance),
                max_steps=self.config.render_max_steps,
            )

            # Convert rendered depth (-1 = miss) to numpy; set misses to 0.
            rendered_np = rendered.cpu().numpy()
            rendered_np[rendered_np < 0] = 0.0
            outputs.depth.send(DepthFrame.new(data=rendered_np, is_metric=True))

            # Periodic maintenance.
            if (
                self.config.decay_every > 0
                and self._frame_count % self.config.decay_every == 0
            ):
                self._mapper.decay()

            if self._frame_count % 10 == 0:
                self._check_vram()

            # Mesh viewer update.
            if (
                self._viewer is not None
                and self._frame_count % self.config.mesh_update_every == 0
            ):
                self._mapper.update_color_mesh()
                mesh = self._mapper.get_color_mesh()
                self._viewer.update(mesh.to_open3d())

        if self._viewer is not None:
            self._viewer.close()
        print("[TSDFMapper] Stopped")
