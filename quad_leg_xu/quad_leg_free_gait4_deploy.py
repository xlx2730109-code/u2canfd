"""Stable entry point for the Bennett free-gait (33-dim, no gait terms) policy.

Same interface as `quad_leg_free_gait3_deploy.py`, but for the free_gait4
experiment. free_gait4 is observation-identical to free_gait3 (the only change is
a sim-side swing-clearance reward term), so it runs through the identical
`--policy-kind free` contract / observation-builder / MIT-position decode. The
launcher auto-selects the newest free_gait4 export, falling back to free_gait3
until free_gait4 has been trained and exported. Pass the runtime flags on the
command line, e.g.:

  D:\\Conda\\envs\\env_isaaclab\\python.exe -B quad_leg_xu\\quad_leg_free_gait4_deploy.py \\
      --keyboard --imu_port COM6 --output_scale 0.5 --duration_s 15
"""

import sys
from pathlib import Path

from bennett_deploy.runtime import main

# free_gait4 first, then free_gait3 (obs-identical) as a fallback. The wildcard
# must stay in the *pattern* string (Path.glob does not expand '*' in the base).
FREE_GAIT_ROOTS = (
    Path(r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_free_gait\quad_leg_free_gait4\flat"),
    Path(r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_free_gait\quad_leg_free_gait3\flat"),
)


def _newest_free_gait_policy() -> Path:
    candidates = sorted(
        (p for root in FREE_GAIT_ROOTS if root.exists()
         for p in root.glob("*/exported/policy.pt")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            "No trained free-gait policy found. Train free_gait4 first, e.g.:\n"
            "  python scripts/reinforcement_learning/rsl_rl_train.py "
            "--task Isaac-BennettRL-Flat-QuadLeg-FreeGait4-v0 --num_envs 4096 "
            "--headless --max_iterations 3000"
        )
    return candidates[0]


if __name__ == "__main__":
    argv = ["--policy-kind", "free", "--policy", str(_newest_free_gait_policy())]
    argv.extend(sys.argv[1:])
    raise SystemExit(main(argv))
