"""Bennett Go2-13 hardware deployment entry point.

This task keeps the verified Go2-11 command-clock and low-latency hardware
runtime, but loads the exact Go2-13 run and mirrors its absolute joint-target
limits.  ``--check_only`` validates the policy/config contract without opening
the IMU, USB-CAN adapter, or motors.
"""

# Run from E:\HuanCun\Desktop\u2canfd:
# D:\Conda\envs\env_isaaclab\python.exe -B .\quad_leg_xu\quad_leg_go2-13.py --check_only
# D:\Conda\envs\env_isaaclab\python.exe -B .\quad_leg_xu\quad_leg_go2-13.py --keyboard --log_csv ".\go2-13-first.csv"
# python .\quad_leg_xu\quad_leg_go2-13.py --keyboard --log_csv ".\go2-13-first.csv"
from __future__ import annotations

import math
import sys
from typing import Sequence

from bennett_deploy import contract as contract_module
from bennett_deploy import policy as policy_module
from bennett_deploy import runtime


DEFAULT_POLICY = (
    r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_go2\quad_leg_go2-13"
    r"\flat\2026-07-27_17-54-37\exported\policy.pt"
)

EXPECTED_PHASE_FUNCTIONS = {
    "observations.policy.crawl_phase.func": (
        "bennett_rl.tasks.manager_based.quad_leg_go2.quad_leg_go2-13.mdp.observations:"
        "commanded_crawl_global_phase_sin_cos"
    ),
    "observations.policy.crawl_leg_phase.func": (
        "bennett_rl.tasks.manager_based.quad_leg_go2.quad_leg_go2-13.mdp.observations:"
        "commanded_crawl_leg_phase_sin_cos"
    ),
    "observations.policy.desired_contacts.func": (
        "bennett_rl.tasks.manager_based.quad_leg_go2.quad_leg_go2-13.mdp.observations:"
        "commanded_crawl_desired_contacts"
    ),
    "observations.policy.gait_params.func": (
        "bennett_rl.tasks.manager_based.quad_leg_go2.quad_leg_go2-13.mdp.observations:"
        "commanded_crawl_gait_params"
    ),
}

EXPECTED_COMMAND_MINIMA = {
    "commands.base_velocity.min_abs_lin_vel_x": 0.06,
    "commands.base_velocity.min_abs_lin_vel_y": 0.04,
    "commands.base_velocity.min_abs_ang_vel_z": 0.15,
}

EXPECTED_TARGET_LIMITS = {
    "actions.joint_pos.clip..*_thigh": (-0.80, 0.80),
    "actions.joint_pos.clip..*_calf": (-0.90, 0.55),
}


class Go213DeploymentContract(contract_module.DeploymentContract):
    """Reject policies whose saved config is not the Go2-13 deployment contract."""

    @classmethod
    def load(cls, policy_arg, *, action_clip_override=None):
        contract = super().load(policy_arg, action_clip_override=action_clip_override)
        env = contract_module.parse_isaaclab_yaml(contract.env_yaml)

        for key, expected in EXPECTED_PHASE_FUNCTIONS.items():
            actual = env.get(key)
            if actual != expected:
                raise ValueError(f"Go2-13 deployment requires {key}={expected!r}, got {actual!r}")

        for key, expected in EXPECTED_COMMAND_MINIMA.items():
            actual = env.get(key)
            if actual is None or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(f"Go2-13 deployment requires {key}={expected}, got {actual!r}")

        for key, expected in EXPECTED_TARGET_LIMITS.items():
            actual = env.get(key)
            if not isinstance(actual, list) or len(actual) != 2:
                raise ValueError(f"Go2-13 deployment requires two target limits at {key}, got {actual!r}")
            if not all(
                math.isclose(float(value), limit, rel_tol=0.0, abs_tol=1.0e-9)
                for value, limit in zip(actual, expected)
            ):
                raise ValueError(f"Go2-13 deployment requires {key}={expected!r}, got {actual!r}")
        return contract


class CommandPhaseClock:
    """Mirror training: reset on stop and advance only while a command is active."""

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


class Go213ObservationBuilder(policy_module.ObservationBuilder):
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


class Go213PolicyPipeline(policy_module.PolicyPipeline):
    """Use the verified policy path and reproduce Go2-13's target clipping."""

    def __init__(self, contract, *, output_scale: float, target_rate_limit_deg_s: float):
        super().__init__(
            contract,
            output_scale=output_scale,
            target_rate_limit_deg_s=target_rate_limit_deg_s,
        )
        self.observations = Go213ObservationBuilder(contract, self.torch)
        self.target_lower = self.torch.tensor(
            (-0.80, -0.90, -0.80, -0.90, -0.80, -0.90, -0.80, -0.90),
            dtype=self.torch.float32,
        )
        self.target_upper = self.torch.tensor(
            (0.80, 0.55, 0.80, 0.55, 0.80, 0.55, 0.80, 0.55),
            dtype=self.torch.float32,
        )

    def reset(self) -> None:
        super().reset()
        self.observations.reset()

    def infer(self, **kwargs):
        step = super().infer(**kwargs)
        target = self.torch.maximum(self.target_lower, self.torch.minimum(self.target_upper, step.target_sim_rad))
        # Keep the rate-limiter state consistent with the command actually sent.
        self.applied_offset = target - self.train_default
        return policy_module.PolicyStep(
            observation=step.observation,
            raw_action=step.raw_action,
            action=step.action,
            desired_offset_rad=step.desired_offset_rad,
            applied_offset_rad=self.applied_offset.clone(),
            target_sim_rad=target,
            gait=step.gait,
        )


def _add_go213_defaults(argv: list[str]) -> None:
    defaults = {
        "--keyboard_step_x": "0.02",
        "--keyboard_min_nonzero_x": "0.06",
        "--keyboard_min_x": "-0.18",
        "--keyboard_max_x": "0.18",
        "--keyboard_step_y": "0.04",
        "--keyboard_step_yaw": "0.15",
        "--control_rate_hz": "250.0",
        "--dm_sdk_profile": "low_latency",
        "--motor_feedback_timeout_s": "0.10",
        # The trained policy has no 120 deg/s target filter.  Hardware telemetry
        # showed that 120 deg/s changed reverse commands by 17.8--20.6 deg p95
        # and made the stance response soft.  360 deg/s preserves the policy's
        # nominal dynamics while still rejecting rare discontinuous jumps.
        "--target_rate_limit_deg_s": "360.0",
    }
    present = {item.split("=", 1)[0] for item in argv[1:] if item.startswith("--")}
    for flag, value in defaults.items():
        if flag not in present:
            argv.extend((flag, value))


def main() -> int:
    _add_go213_defaults(sys.argv)
    runtime.DEFAULT_POLICY = DEFAULT_POLICY
    runtime.DeploymentContract = Go213DeploymentContract
    runtime.PolicyPipeline = Go213PolicyPipeline
    return runtime.main()


if __name__ == "__main__":
    raise SystemExit(main())
