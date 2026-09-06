"""Stable entry point for the Bennett free-gait (33-dim, no gait terms) policy.

Bakes the free-gait contract (`--policy-kind free`) and auto-selects the newest
trained `quad_leg_free_gait3` exported policy, so the only flags needed at the
console are the runtime ones (e.g. --keyboard --imu_port COM6). Pass these on
the command line to add/override, e.g.:

  D:\\Conda\\envs\\env_isaaclab\\python.exe -B quad_leg_xu\\quad_leg_free_gait3_deploy.py \\
      --keyboard --imu_port COM6 --output_scale 0.5 --duration_s 15
"""

import sys
from pathlib import Path

from bennett_deploy.runtime import main

FREE_GAIT_GLOB = (
    Path(r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_free_gait\quad_leg_free_gait3\flat")
)


def _newest_free_gait_policy() -> Path:
    # Path.glob does not expand wildcards in the *base* path, so the wildcard
    # must appear in the pattern string, not in the base Path.
    candidates = sorted(FREE_GAIT_GLOB.glob("*/exported/policy.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(
            "No trained quad_leg_free_gait3 policy found. Train it first, e.g.:\n"
            "  python scripts/reinforcement_learning/rsl_rl_train.py "
            "--task Isaac-BennettRL-Flat-QuadLeg-FreeGait3-v0 --num_envs 4096 "
            "--headless --max_iterations 3000"
        )
    return candidates[0]


if __name__ == "__main__":
    argv = ["--policy-kind", "free", "--policy", str(_newest_free_gait_policy())]
    argv.extend(sys.argv[1:])
    raise SystemExit(main(argv))
