"""Portable paths for the Bennett training repository used by deployment scripts."""

import os
from pathlib import Path


def bennett_root() -> Path:
    override = os.environ.get("BENNETT_RL_ROOT")
    if override:
        return Path(override).expanduser()

    # Ubuntu checkout: Robot_Project/{Hardware_Debug/u2canfd,bennett_rl}.
    sibling_checkout = Path(__file__).resolve().parents[4] / "bennett_rl"
    # Preserve the existing Windows checkout location when it is present.
    windows_checkout = Path("E:/Project/Isaaclab/bennett_rl")
    for candidate in (sibling_checkout, windows_checkout):
        if candidate.is_dir():
            return candidate
    return windows_checkout if os.name == "nt" else sibling_checkout


def bennett_path(*parts: str) -> str:
    return str(bennett_root().joinpath(*parts))
