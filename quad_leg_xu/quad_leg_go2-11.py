"""Bennett Go2-11 hardware entry point (offline candidate).

This adapter deliberately reuses the verified Bennett motor/safety runtime but
replaces the Go2-10 episode-time gait clock with Go2-11's command-state clock.
No hardware is opened by ``--check_only``.
"""

# D:\Conda\envs\env_isaaclab\python.exe -B quad_leg_xu\quad_leg_go2-11.py --keyboard

from __future__ import annotations

import math
import sys
from typing import Sequence

from bennett_deploy import contract as contract_module
from bennett_deploy import policy as policy_module
from bennett_deploy import runtime


DEFAULT_POLICY = (
    r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_go2\quad_leg_go2-11"
    r"\flat\2026-07-20_03-10-22\exported\policy.pt"
)

EXPECTED_PHASE_FUNCTIONS = {
    "observations.policy.crawl_phase.func": (
        "bennett_rl.tasks.manager_based.quad_leg_go2.quad_leg_go2-11.mdp.observations:"
        "commanded_crawl_global_phase_sin_cos"
    ),
    "observations.policy.crawl_leg_phase.func": (
        "bennett_rl.tasks.manager_based.quad_leg_go2.quad_leg_go2-11.mdp.observations:"
        "commanded_crawl_leg_phase_sin_cos"
    ),
    "observations.policy.desired_contacts.func": (
        "bennett_rl.tasks.manager_based.quad_leg_go2.quad_leg_go2-11.mdp.observations:"
        "commanded_crawl_desired_contacts"
    ),
    "observations.policy.gait_params.func": (
        "bennett_rl.tasks.manager_based.quad_leg_go2.quad_leg_go2-11.mdp.observations:"
        "commanded_crawl_gait_params"
    ),
}


class Go211DeploymentContract(contract_module.DeploymentContract):
    """Reject a policy unless its saved config has Go2-11 phase semantics."""

    @classmethod
    def load(cls, policy_arg, *, action_clip_override=None):
        contract = super().load(policy_arg, action_clip_override=action_clip_override)
        env = contract_module.parse_isaaclab_yaml(contract.env_yaml)
        for key, expected in EXPECTED_PHASE_FUNCTIONS.items():
            actual = env.get(key)
            if actual != expected:
                raise ValueError(f"Go2-11 deployment requires {key}={expected!r}, got {actual!r}")
        expected_minima = {
            "commands.base_velocity.min_abs_lin_vel_x": 0.06,
            "commands.base_velocity.min_abs_lin_vel_y": 0.04,
            "commands.base_velocity.min_abs_ang_vel_z": 0.15,
        }
        for key, expected in expected_minima.items():
            actual = env.get(key)
            if actual is None or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(f"Go2-11 deployment requires {key}={expected}, got {actual!r}")
        return contract


class CommandPhaseClock:
    """Mirror Go2-11: start at zero, advance only while commanded, reset on stop."""

    def __init__(self, *, dt: float, frequency_hz: float, command_deadband: float):
        self.dt = float(dt)
        self.frequency_hz = float(frequency_hz)
        self.command_deadband = float(command_deadband)
        self.phase = 0.0
        self.was_moving = False

    def reset(self) -> None:
        self.phase = 0.0
        self.was_moving = False

    def update(self, command: Sequence[float]) -> float:
        if len(command) != 3:
            raise ValueError("velocity command must contain vx, vy, yaw")
        moving = math.sqrt(sum(float(value) ** 2 for value in command)) >= self.command_deadband
        if not moving:
            self.phase = 0.0
            self.was_moving = False
        elif not self.was_moving:
            self.phase = 0.0
            self.was_moving = True
        else:
            self.phase = (self.phase + self.dt * self.frequency_hz) % 1.0
        return self.phase


class Go211ObservationBuilder(policy_module.ObservationBuilder):
    def __init__(self, contract, torch_module):
        super().__init__(contract, torch_module)
        self.clock = CommandPhaseClock(
            dt=1.0 / contract.policy_rate_hz,
            frequency_hz=contract.gait_frequency_hz,
            command_deadband=contract.command_deadband,
        )

    def reset(self) -> None:
        self.clock.reset()

    def build(self, *, velocity_command, phase_elapsed_s, **kwargs):
        del phase_elapsed_s
        phase = self.clock.update(velocity_command)
        return super().build(
            velocity_command=velocity_command,
            phase_elapsed_s=phase / self.contract.gait_frequency_hz,
            **kwargs,
        )


class Go211PolicyPipeline(policy_module.PolicyPipeline):
    def __init__(self, contract, *, output_scale: float, target_rate_limit_deg_s: float):
        super().__init__(
            contract,
            output_scale=output_scale,
            target_rate_limit_deg_s=target_rate_limit_deg_s,
        )
        self.observations = Go211ObservationBuilder(contract, self.torch)

    def reset(self) -> None:
        super().reset()
        if hasattr(self.observations, "reset"):
            self.observations.reset()


def _add_go211_defaults(argv: list[str]) -> None:
    defaults = {
        # Training sampled non-zero vx only in +/-[0.06, 0.18]m/s.  The
        # keyboard therefore jumps 0 -> +/-0.06, then uses 0.02m/s fine steps.
        "--keyboard_step_x": "0.02",
        "--keyboard_min_nonzero_x": "0.06",
        "--keyboard_min_x": "-0.18",
        "--keyboard_max_x": "0.18",
        "--keyboard_step_y": "0.04",
        "--keyboard_step_yaw": "0.15",
        # The verified low-latency Windows path holds feedback age to <=16ms;
        # 250Hz sends five motor targets per 50Hz policy period.
        "--control_rate_hz": "250.0",
        # Only Go2-11 opts into the isolated, hash-verified SDK patch.  The
        # shared runtime and all other entry points keep the original default.
        "--dm_sdk_profile": "low_latency",
        # Go2-11 low-latency feedback measured <=31 ms worst-case in the
        # walking log; 100 ms keeps margin while failing much faster than the
        # shared runtime's conservative 500 ms default.
        "--motor_feedback_timeout_s": "0.10",
        # The unfiltered policy changed RR_calf by 8.92 deg p95 and up to
        # 35.04 deg between logged policy samples.  At 120 deg/s, each 20 ms
        # policy update can move at most 2.4 deg while retaining full amplitude.
        "--target_rate_limit_deg_s": "120.0",
    }
    present = {item.split("=", 1)[0] for item in argv[1:] if item.startswith("--")}
    for flag, value in defaults.items():
        if flag not in present:
            argv.extend((flag, value))


def main() -> int:
    _add_go211_defaults(sys.argv)
    runtime.DEFAULT_POLICY = DEFAULT_POLICY
    runtime.DeploymentContract = Go211DeploymentContract
    runtime.PolicyPipeline = Go211PolicyPipeline
    return runtime.main()


if __name__ == "__main__":
    raise SystemExit(main())
