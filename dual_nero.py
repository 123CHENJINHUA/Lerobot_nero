"""Two Nero leader/follower pairs, each on its own CAN bus."""

from dataclasses import dataclass, replace

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot_robot_nero import NeroFollower, NeroFollowerConfig
from lerobot_teleoperator_nero import NeroLeader, NeroLeaderConfig
from nero_common import DUAL_FEATURES, FEATURES, SIDES


@NeroFollowerConfig.register_subclass("nero_dual_follower")
@dataclass(kw_only=True)
class NeroDualFollowerConfig(NeroFollowerConfig):
    left_channel: str = "can0"
    right_channel: str = "can1"

    def __post_init__(self):
        super().__post_init__()
        if self.left_channel == self.right_channel:
            raise ValueError("Left and right arms require separate CAN buses")
        if self.action_space != "joints":
            raise ValueError("Dual Nero currently supports joint actions only")


@NeroLeaderConfig.register_subclass("nero_dual_leader")
@dataclass(kw_only=True)
class NeroDualLeaderConfig(NeroLeaderConfig):
    left_channel: str = "can0"
    right_channel: str = "can1"

    def __post_init__(self):
        super().__post_init__()
        if self.left_channel == self.right_channel:
            raise ValueError("Left and right arms require separate CAN buses")


class NeroDualFollower(NeroFollower):
    config_class = NeroDualFollowerConfig
    name = "nero_dual_follower"

    def __init__(self, config):
        super().__init__(config)
        self.arms = {}
        for side in SIDES:
            arm = NeroFollower(replace(config, channel=getattr(config, f"{side}_channel"),
                                       id=f"{config.id}_{side}", cameras={}))
            arm.depth_hook = None
            self.arms[side] = arm

    @property
    def action_features(self):
        return dict.fromkeys(DUAL_FEATURES, float)

    @property
    def observation_features(self):
        return {**{f"{side}.{key}": value for side, arm in self.arms.items()
                   for key, value in arm.observation_features.items()},
                **{name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()},
                **(self.depth_hook.image_features if self.depth_hook is not None else {})}

    @property
    def is_connected(self):
        return all(arm.is_connected for arm in self.arms.values()) and all(
            camera.is_connected for camera in self.cameras.values())

    def connect(self, calibrate=True):
        if any(arm.is_connected for arm in self.arms.values()):
            raise RuntimeError("Dual Nero already connected")
        try:
            for arm in self.arms.values():
                arm.connect(calibrate)
            for camera in self.cameras.values():
                camera.connect()
            if self.depth_hook is not None:
                self.depth_hook.attach()
        except BaseException:
            self.disconnect()
            raise

    def get_observation(self):
        if not self.is_connected:
            raise RuntimeError("Dual Nero is not connected")
        # shortcut: observations share a loop tick, use hardware triggering for exposure synchronization.
        obs = {f"{side}.{key}": value for side, arm in self.arms.items()
               for key, value in arm.get_observation().items()}
        obs.update({name: camera.async_read() for name, camera in self.cameras.items()})
        if self.depth_hook is not None:
            obs.update(self.depth_hook.snapshot_images())
        return obs

    def send_action(self, action):
        if not self.is_connected:
            raise RuntimeError("Dual Nero is not connected")
        if set(action) != set(DUAL_FEATURES):
            raise ValueError(f"Expected dual action keys: {DUAL_FEATURES}")
        if self.config.control_mode == "external":
            return {f"{side}.{key}": value for side, arm in self.arms.items()
                    for key, value in arm.send_action(
                        {key: action[f"{side}.{key}"] for key in FEATURES}).items()}
        if not self.config.allow_motion:
            raise RuntimeError("Host motion is disabled; set --robot.allow_motion=true explicitly")
        # Validate both targets before sending either arm a command.
        prepared = {side: arm._prepare_joint_action([float(action[f"{side}.{key}"]) for key in FEATURES])
                    for side, arm in self.arms.items()}
        for side, arm in self.arms.items():
            arm._commit_joint_action(prepared[side])
        return {f"{side}.{key}": value for side in SIDES
                for key, value in zip(FEATURES, prepared[side], strict=True)}

    def disconnect(self):
        try:
            super().disconnect()
        finally:
            try:
                self.arms["left"].disconnect()
            finally:
                self.arms["right"].disconnect()


class NeroDualLeader(NeroLeader):
    config_class = NeroDualLeaderConfig
    name = "nero_dual_leader"

    def __init__(self, config):
        Teleoperator.__init__(self, config)
        self.arms = {side: NeroLeader(replace(config, channel=getattr(config, f"{side}_channel"),
                                             id=f"{config.id}_{side}")) for side in SIDES}

    @property
    def action_features(self):
        return dict.fromkeys(DUAL_FEATURES, float)

    @property
    def is_connected(self):
        return all(arm.is_connected for arm in self.arms.values())

    def connect(self, calibrate=True):
        try:
            for arm in self.arms.values():
                arm.connect(calibrate)
        except BaseException:
            self.disconnect()
            raise

    def get_action(self):
        return {f"{side}.{key}": value for side, arm in self.arms.items()
                for key, value in arm.get_action().items()}

    def disconnect(self):
        try:
            self.arms["left"].disconnect()
        finally:
            self.arms["right"].disconnect()
