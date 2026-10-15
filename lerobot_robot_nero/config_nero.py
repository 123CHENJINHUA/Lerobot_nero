from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

from nero_common import ROOT


@RobotConfig.register_subclass("nero_follower")
@dataclass(kw_only=True)
class NeroFollowerConfig(RobotConfig):
    channel: str = "can0"
    firmware: str = "v121"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    allow_motion: bool = False
    manage_mode: bool = False
    control_mode: Literal["external", "host"] = "external"
    max_feedback_age_s: float = 0.5
    max_initial_error_rad: float = 0.25
    max_step_rad: float = 0.06
    max_gripper_step_m: float = 0.006
    gripper_max_m: float = 0.08
    gripper_force_n: float = 1.0
    calibration_dir: Path | None = ROOT / "calibration" / "robots"

    def __post_init__(self):
        super().__post_init__()
        if self.id is None:
            self.id = self.channel
        if not 0.1 <= self.max_feedback_age_s <= 2:
            raise ValueError("max_feedback_age_s must be 0.1..2")
        if not 0 < self.max_step_rad <= 0.2:
            raise ValueError("max_step_rad must be 0..0.2")
        if not 0 < self.max_gripper_step_m <= 0.02:
            raise ValueError("max_gripper_step_m must be 0..0.02")
