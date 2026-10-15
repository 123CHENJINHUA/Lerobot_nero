from dataclasses import dataclass
from pathlib import Path

from lerobot.teleoperators.config import TeleoperatorConfig

from nero_common import ROOT


@TeleoperatorConfig.register_subclass("nero_leader")
@dataclass(kw_only=True)
class NeroLeaderConfig(TeleoperatorConfig):
    channel: str = "can0"
    firmware: str = "v121"
    max_feedback_age_s: float = 0.5
    calibration_dir: Path | None = ROOT / "calibration" / "teleoperators"

    def __post_init__(self):
        if self.id is None:
            self.id = self.channel
        if not 0.1 <= self.max_feedback_age_s <= 2:
            raise ValueError("max_feedback_age_s must be 0.1..2")
