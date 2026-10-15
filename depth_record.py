"""Expose compact D435i depth views as LeRobot visual observations."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from camera_serial import depth_device_for_serial

ACTIVE_HOOK = None


class DepthStream:
    """Read a RealSense Z16 V4L2 stream without requiring pyrealsense2."""

    def __init__(self, serial: str, width: int, height: int, fps: int):
        self.serial = serial
        self.device = depth_device_for_serial(serial)
        self.width, self.height = width, height
        self.frame_bytes = width * height * 2
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.process = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "v4l2",
             "-input_format", "gray16le", "-video_size", f"{width}x{height}",
             "-framerate", str(fps), "-i", self.device, "-f", "rawvideo",
             "-pix_fmt", "gray16le", "pipe:1"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.thread = threading.Thread(target=self._reader, name=f"depth-{serial}", daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            with self.lock:
                if self.latest is not None:
                    return
                error = self.error
            if error is not None:
                self.close()
                raise RuntimeError(f"Depth camera {serial} failed: {error}")
            time.sleep(0.02)
        self.close()
        raise RuntimeError(f"No Z16 depth frames from camera {serial} within 8 s")

    def _reader(self):
        try:
            while True:
                raw = self.process.stdout.read(self.frame_bytes)
                if len(raw) != self.frame_bytes:
                    break
                frame = np.frombuffer(raw, dtype="<u2").reshape(self.height, self.width).copy()
                with self.lock:
                    self.latest = (frame, time.monotonic())
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            if self.process.poll() is not None:
                detail = self.process.stderr.read().decode(errors="replace").strip()
                with self.lock:
                    self.error = detail or f"ffmpeg exited {self.process.returncode}"

    def snapshot(self, max_age_s: float = 0.5) -> np.ndarray:
        with self.lock:
            latest, error = self.latest, self.error
        if latest is None or time.monotonic() - latest[1] > max_age_s:
            raise RuntimeError(f"Depth camera {self.serial} has no fresh frame: {error}")
        return latest[0]

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.thread.join(timeout=2)
        self.process.stdout.close()
        self.process.stderr.close()


def depth_to_rgb(frame: np.ndarray, near_raw: int, far_raw: int) -> np.ndarray:
    """Map a fixed Z16 range to an 8-bit, three-channel depth view.

    Code 0 means invalid depth. Valid values use 1..255; values outside the
    configured range are clipped. The fixed mapping is shared by all episodes.
    """
    if frame.dtype != np.uint16 or frame.ndim != 2:
        raise ValueError(f"Depth frame must be uint16 HxW, got {frame.dtype} {frame.shape}")
    values = np.clip(frame, near_raw, far_raw).astype(np.float32)
    gray = np.rint(1 + (values - near_raw) * (254 / (far_raw - near_raw))).astype(np.uint8)
    gray[frame == 0] = 0
    return np.repeat(gray[..., None], 3, axis=2)


class DepthObservation:
    def __init__(self, root: Path | None, cameras: dict[str, str], width: int, height: int, fps: int,
                 near_raw: int, far_raw: int):
        self.root = root
        self.cameras = cameras
        self.width, self.height, self.fps = width, height, fps
        self.near_raw, self.far_raw = near_raw, far_raw
        self.streams: dict[str, DepthStream] = {}

    @property
    def image_features(self):
        return {f"{name}_depth": (self.height, self.width, 3) for name in self.cameras}

    def attach(self):
        if self.root is not None:
            metadata_path = self.root / "meta" / "nero_depth_encoding.json"
            metadata = {
                "source": "RealSense Z16 raw depth", "storage": "LeRobot H.264 three-channel 8-bit video",
                "near_raw": self.near_raw, "far_raw": self.far_raw,
                "invalid_code": 0, "valid_code_range": [1, 255],
                "conversion": "clip(raw, near_raw, far_raw) -> round(1 + 254 * (raw - near_raw) / (far_raw - near_raw)); raw=0 -> 0",
                "width": self.width, "height": self.height, "fps": self.fps,
                "cameras": {f"{name}_depth": serial for name, serial in self.cameras.items()},
                "note": "H.264 is lossy; decoded pixels cannot recover exact raw depth or invalid mask",
            }
            if metadata_path.exists():
                if json.loads(metadata_path.read_text()) != metadata:
                    raise ValueError(f"Depth settings differ from the existing dataset: {metadata_path}")
            else:
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        try:
            for name, serial in self.cameras.items():
                self.streams[name] = DepthStream(serial, self.width, self.height, self.fps)
        except BaseException:
            self.detach()
            raise

    def detach(self):
        for stream in self.streams.values():
            stream.close()
        self.streams.clear()

    def snapshot_images(self):
        return {
            f"{name}_depth": depth_to_rgb(stream.snapshot(), self.near_raw, self.far_raw)
            for name, stream in self.streams.items()
        }


def install(root: Path | None, cameras: dict[str, str], width: int, height: int, fps: int,
            near_raw: int, far_raw: int):
    global ACTIVE_HOOK
    if ACTIVE_HOOK is not None:
        raise RuntimeError("Depth observation is already installed")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to capture D435i depth frames")
    if not 0 <= near_raw < far_raw <= 65535:
        raise ValueError("Depth raw range must satisfy 0 <= near < far <= 65535")
    ACTIVE_HOOK = DepthObservation(root, cameras, width, height, fps, near_raw, far_raw)
