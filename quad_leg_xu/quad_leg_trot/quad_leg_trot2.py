"""Bennett Trot2 hardware deployment for motor-model A/B variant B.

Trot2 is the trained ``8 Nm / 20 Nm / 19.896753 rad/s`` motor-B policy.
It reuses Trot1's exact speed-conditioned diagonal clock, 50-value policy
observation, trained joint limits, and the verified low-latency hardware
runtime.  ``--check_only`` never opens hardware.
"""

# Run from E:\HuanCun\Desktop\u2canfd:
# D:\Conda\envs\env_isaaclab\python.exe -B .\quad_leg_xu\quad_leg_trot\quad_leg_trot2.py --check_only
# D:\Conda\envs\env_isaaclab\python.exe -B .\quad_leg_xu\quad_leg_trot\quad_leg_trot2.py --keyboard --log_csv ".\quad_leg_xu\quad_leg_trot\logs\trot2-first.csv"
# python .\quad_leg_xu\quad_leg_trot\quad_leg_trot2.py --keyboard --log_csv ".\quad_leg_xu\quad_leg_trot\logs\trot2-first.csv"
from __future__ import annotations

import math
import sys
from pathlib import Path


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from bennett_deploy import runtime
from quad_leg_trot1 import (
    Trot1DeploymentContract,
    Trot1DeploymentRunner,
    Trot1PolicyPipeline,
    _add_trot1_defaults,
)


DEFAULT_POLICY = (
    r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_trot"
    r"\quad_leg_trot-motor-ab\b_datasheet\2026-07-28_22-53-51"
    r"\exported\policy.pt"
)

EXPECTED_MOTOR_B = {
    "effort_limit": 8.0,
    "saturation_effort": 20.0,
    "velocity_limit": 19.8967534727,
}


class Trot2DeploymentContract(Trot1DeploymentContract):
    """Add a hard motor-B identity check to the full Trot1 contract."""

    @classmethod
    def load(cls, policy_arg, *, action_clip_override=None):
        contract = super().load(
            policy_arg,
            action_clip_override=action_clip_override,
        )
        for field, expected in EXPECTED_MOTOR_B.items():
            actual = float(getattr(contract, field))
            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                raise ValueError(
                    f"Trot2 requires motor-B {field}={expected}, got {actual}"
                )
        return contract


class Trot2DeploymentRunner(Trot1DeploymentRunner):
    def print_contract(self) -> None:
        super().print_contract()
        print(
            "[MOTOR-B] verified effort=8.000Nm saturation=20.000Nm "
            "velocity=19.896753rad/s"
        )


def main() -> int:
    _add_trot1_defaults(sys.argv)
    runtime.DEFAULT_POLICY = DEFAULT_POLICY
    runtime.DeploymentContract = Trot2DeploymentContract
    runtime.PolicyPipeline = Trot1PolicyPipeline
    runtime.DeploymentRunner = Trot2DeploymentRunner
    return runtime.main()


if __name__ == "__main__":
    raise SystemExit(main())
