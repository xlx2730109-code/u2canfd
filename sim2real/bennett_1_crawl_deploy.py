"""Hardware entry point for the bennett_1 crawl-variant policy (50 -> 8, clocked static crawl).

Default policy: T9 (`logs/rsl_rl/bennett_1/crawl/2026-10-16_13-10-20_t9_same`), the policy the sim2sim config
`bennett_1_crawl` loads. Use another run with the environment variable `BENNETT1_CRAWL_RUN=<run directory name>`
or with `--policy <exported/policy.pt>`. The run directory must contain `params/env.yaml`, `params/agent.yaml`
and `exported/policy.pt` (export with `scripts/rsl_rl/play.py --headless`).

Keyboard command ranges default to the training ranges (vx +/-0.25, vy +/-0.12, yaw +/-0.5); flags given on the
command line override them. Examples (from the sim2real directory):

  python bennett_1_crawl_deploy.py --check_only          # no IMU, no motors
  python bennett_1_crawl_deploy.py --dry_run --keyboard  # IMU only, no motors
"""

import os
import sys

from bennett_deploy.paths import bennett_path
from bennett_deploy.runtime import main

DEFAULT_RUN = "2026-10-16_13-10-20_t9_same"
TRAIN_RANGES = [
    "--keyboard_min_x", "-0.25", "--keyboard_max_x", "0.25",
    "--keyboard_min_y", "-0.12", "--keyboard_max_y", "0.12",
    "--keyboard_min_yaw", "-0.5", "--keyboard_max_yaw", "0.5",
]


def default_policy() -> str:
    run = os.environ.get("BENNETT1_CRAWL_RUN", DEFAULT_RUN)
    return bennett_path("logs", "rsl_rl", "bennett_1", "crawl", run, "exported", "policy.pt")


if __name__ == "__main__":
    argv = ["--policy-kind", "go2", "--policy", default_policy(), *TRAIN_RANGES]
    argv.extend(sys.argv[1:])
    raise SystemExit(main(argv))
