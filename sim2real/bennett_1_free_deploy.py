"""Hardware entry point for the bennett_1 free-gait policy (33 -> 8, no gait clock).

Default policy: S1, the policy the sim2sim config `bennett_1_free` loads. S1 is bundled in
`policies/bennett_1_free_S1`, so no bennett_rl checkout is needed. Select with the environment variable
`BENNETT1_FREE_RUN=S1`, a full run directory name under `bennett_rl/logs/rsl_rl/bennett_1/flat`, or
`--policy <exported/policy.pt>`. A run directory must contain `params/env.yaml`, `params/agent.yaml` and
`exported/policy.pt` (export with `scripts/rsl_rl/play.py --headless`).

Keyboard command ranges default to the training ranges (vx +/-0.35, vy +/-0.20, yaw +/-0.6); flags given on the
command line override them. Examples (from the sim2real directory):

  python bennett_1_free_deploy.py --check_only          # no IMU, no motors
  python bennett_1_free_deploy.py --dry_run --keyboard  # IMU only, no motors
"""

import os
import sys

from bennett_deploy.paths import resolve_policy
from bennett_deploy.runtime import main

DEFAULT_RUN = "S1"
BUNDLED = {"S1": "2026-10-16_05-53-35"}
TRAIN_RANGES = [
    "--keyboard_min_x", "-0.35", "--keyboard_max_x", "0.35",
    "--keyboard_min_y", "-0.20", "--keyboard_max_y", "0.20",
    "--keyboard_min_yaw", "-0.6", "--keyboard_max_yaw", "0.6",
]


def default_policy() -> str:
    run = os.environ.get("BENNETT1_FREE_RUN", DEFAULT_RUN)
    local = f"bennett_1_free_{run}" if run in BUNDLED else None
    return resolve_policy(local, "bennett_1", "flat", BUNDLED.get(run, run), "exported", "policy.pt")


if __name__ == "__main__":
    argv = ["--policy-kind", "free", "--policy", default_policy(), *TRAIN_RANGES]
    argv.extend(sys.argv[1:])
    raise SystemExit(main(argv))
