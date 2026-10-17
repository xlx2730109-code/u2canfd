"""Portable paths for the Bennett training repository used by deployment scripts."""

import os
from pathlib import Path


def bennett_root() -> Path:
    """Locate the bennett_rl checkout.

    Order: environment variable ``BENNETT_RL_ROOT``; a sibling checkout
    (Ubuntu layout ``Robot_Project/{Hardware_Debug/u2canfd, bennett_rl}``);
    ``<drive>/Project/Isaaclab/bennett_rl`` on the drive that holds this
    repository (Windows layout). Set ``BENNETT_RL_ROOT`` for any other layout.
    """

    override = os.environ.get("BENNETT_RL_ROOT")
    if override:
        return Path(override).expanduser()

    here = Path(__file__).resolve()
    sibling_checkout = here.parents[4] / "bennett_rl"
    same_drive_checkout = Path(here.anchor) / "Project" / "Isaaclab" / "bennett_rl"
    for candidate in (sibling_checkout, same_drive_checkout):
        if candidate.is_dir():
            return candidate
    return same_drive_checkout if os.name == "nt" else sibling_checkout


def bennett_path(*parts: str) -> str:
    return str(bennett_root().joinpath(*parts))
