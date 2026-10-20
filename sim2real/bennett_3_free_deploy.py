"""Hardware entry point for the bennett_3 free-gait policy (33 -> 8, no gait clock).

bennett_3 is the bennett_1 (S1) training configuration with the actuator replaced by the identified model
(command delay 5-15 ms, joint friction, armature 0.0043). Default policy: the final checkpoint (model_1200) of
`logs/rsl_rl/bennett/bennett_3/flat/2026-10-20_16-38-16`, bundled in `policies/bennett_3_free_V5`, so no bennett_rl
checkout is needed. Select with the environment variable `BENNETT3_FREE_RUN=<run directory name under
bennett_rl/logs/rsl_rl/bennett/bennett_3/flat>` or with `--policy <exported/policy.pt>`. A run directory must contain
`params/env.yaml`, `params/agent.yaml` and `exported/policy.pt` (export with `scripts/rsl_rl/play.py --headless`).

Keyboard command ranges default to the training ranges (vx +/-0.35, vy +/-0.20, yaw +/-0.6); flags given on the
command line override them. Examples (from the sim2real directory):

  python bennett_3_free_deploy.py --check_only          # no IMU, no motors
  python bennett_3_free_deploy.py --dry_run --keyboard  # IMU only, no motors
"""

import os
import sys

from bennett_deploy.paths import resolve_policy
from bennett_deploy.runtime import main

BUNDLED_RUN = "2026-10-20_16-38-16"
TRAIN_RANGES = [
    "--keyboard_min_x", "-0.35", "--keyboard_max_x", "0.35",
    "--keyboard_min_y", "-0.20", "--keyboard_max_y", "0.20",
    "--keyboard_min_yaw", "-0.6", "--keyboard_max_yaw", "0.6",
]


def default_policy() -> str:
    run = os.environ.get("BENNETT3_FREE_RUN", BUNDLED_RUN)
    local = "bennett_3_free_V5" if run == BUNDLED_RUN else None
    return resolve_policy(local, "bennett", "bennett_3", "flat", run, "exported", "policy.pt")


if __name__ == "__main__":
    argv = ["--policy-kind", "free", "--policy", default_policy(), *TRAIN_RANGES]
    argv.extend(sys.argv[1:])
    raise SystemExit(main(argv))
