"""Find a RealSense RGB or depth capture device by its USB serial number."""

from __future__ import annotations

import argparse
from pathlib import Path


def device_for_serial(serial: str, stream: str = "rgb", sysfs_root: Path = Path("/sys/class/video4linux")) -> str:
    if not serial or not serial.isdigit():
        raise ValueError(f"RealSense serial number must contain digits only: {serial!r}")
    if stream not in {"rgb", "depth"}:
        raise ValueError(f"Unsupported RealSense stream: {stream!r}")

    matches = []
    for entry in sysfs_root.glob("video[0-9]*"):
        try:
            interface = (entry / "device").resolve(strict=True)
            usb_serial = (interface.parent / "serial").read_text().strip()
            interface_name = (interface / "interface").read_text().strip().upper()
            index = int((entry / "index").read_text().strip())
        except (FileNotFoundError, OSError, ValueError):
            continue
        if usb_serial == serial and "REALSENSE" in interface_name and interface_name.endswith(f" {stream.upper()}") and index == 0:
            matches.append(f"/dev/{entry.name}")

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one RealSense {stream} capture node for serial {serial}, found {len(matches)}. "
            "Check that this camera is connected."
        )
    return matches[0]


def rgb_device_for_serial(serial: str, sysfs_root: Path = Path("/sys/class/video4linux")) -> str:
    return device_for_serial(serial, "rgb", sysfs_root)


def depth_device_for_serial(serial: str, sysfs_root: Path = Path("/sys/class/video4linux")) -> str:
    return device_for_serial(serial, "depth", sysfs_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", help="RealSense USB serial number")
    parser.add_argument("--stream", choices=("rgb", "depth"), default="rgb")
    args = parser.parse_args()
    try:
        print(device_for_serial(args.serial, args.stream))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
