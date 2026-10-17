"""Hardware entry point for the bennett_1 crawl-variant policy (50 -> 8, clocked static crawl).

Default policy: T9, the policy the sim2sim config `bennett_1_crawl` loads. T9 and D4 are bundled in
`policies/bennett_1_crawl_T9` and `policies/bennett_1_crawl_D4`, so no bennett_rl checkout is needed. Select with the
environment variable `BENNETT1_CRAWL_RUN=T9|D4`, a full run directory name under
`bennett_rl/logs/rsl_rl/bennett_1/crawl`, or `--policy <exported/policy.pt>`. A run directory must contain
`params/env.yaml`, `params/agent.yaml` and `exported/policy.pt` (export with `scripts/rsl_rl/play.py --headless`).

Keyboard command ranges default to the training ranges (vx +/-0.25, vy +/-0.12, yaw +/-0.5); flags given on the
command line override them. Examples (from the sim2real directory):

  python bennett_1_crawl_deploy.py --check_only          # no IMU, no motors
  python bennett_1_crawl_deploy.py --dry_run --keyboard  # IMU only, no motors
"""

import os
import sys

from bennett_deploy.paths import resolve_policy
from bennett_deploy.runtime import main

DEFAULT_RUN = "T9"
BUNDLED = {"T9": "2026-10-16_13-10-20_t9_same", "D4": "2026-10-16_15-13-42_d4_delay5"}
TRAIN_RANGES = [
    "--keyboard_min_x", "-0.25", "--keyboard_max_x", "0.25",
    "--keyboard_min_y", "-0.12", "--keyboard_max_y", "0.12",
    "--keyboard_min_yaw", "-0.5", "--keyboard_max_yaw", "0.5",
]


def default_policy() -> str:
    run = os.environ.get("BENNETT1_CRAWL_RUN", DEFAULT_RUN)
    local = f"bennett_1_crawl_{run}" if run in BUNDLED else None
    return resolve_policy(local, "bennett_1", "crawl", BUNDLED.get(run, run), "exported", "policy.pt")


if __name__ == "__main__":
    argv = ["--policy-kind", "go2", "--policy", default_policy(), *TRAIN_RANGES]
    argv.extend(sys.argv[1:])
    raise SystemExit(main(argv))
