"""Stable entry point for the Bennett trot1 gait hardware policy.

Bakes the speed-conditioned trot1 contract (`--policy-kind trot1`) and the
trot1 exported policy so the only flags needed at the console are the runtime
ones (e.g. --keyboard --imu_port COM6). Pass these on the command line to
add/override, e.g.:

  D:\\Conda\\envs\\env_isaaclab\\python.exe -B quad_leg_xu\\quad_leg_trot1_deploy.py \\
      --keyboard --imu_port COM6 --output_scale 0.5 --duration_s 15
"""

import sys

from bennett_deploy.runtime import main

TROT1_POLICY = r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_trot\quad_leg_trot1\flat\2026-07-27_20-08-08\exported"


if __name__ == "__main__":
    argv = ["--policy-kind", "trot1", "--policy", TROT1_POLICY]
    argv.extend(sys.argv[1:])
    raise SystemExit(main(argv))
