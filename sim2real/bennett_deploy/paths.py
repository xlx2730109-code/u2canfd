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


POLICY_DIR = Path(__file__).resolve().parents[1] / "policies"


def local_policy(name: str) -> str | None:
    """Path of ``sim2real/policies/<name>/exported/policy.pt`` if that bundled copy exists."""
    path = POLICY_DIR / name / "exported" / "policy.pt"
    return str(path) if path.is_file() else None


def resolve_policy(local_name: str | None, *repo_parts: str) -> str:
    """Bundled policy copy first; otherwise ``bennett_rl/logs/rsl_rl/<repo_parts>``.

    A bundled copy is a run directory reduced to ``exported/policy.pt``, ``params/env.yaml`` and
    ``params/agent.yaml``, so deployment needs no bennett_rl checkout.
    """
    if local_name:
        local = local_policy(local_name)
        if local:
            return local
    return bennett_path("logs", "rsl_rl", *repo_parts)
