"""Bennett Trot1 hardware deployment entry point.

The adapter reuses the verified low-latency Bennett motor/safety runtime while
reproducing Trot1's speed-conditioned diagonal gait clock and absolute joint
target limits.  ``--check_only`` never opens the IMU, USB-CAN adapter, or
motors.
"""

# Run from E:\HuanCun\Desktop\u2canfd:
# old: Trot1 previously lived directly under quad_leg_xu.
# D:\Conda\envs\env_isaaclab\python.exe -B .\quad_leg_xu\quad_leg_trot\quad_leg_trot1.py --check_only
# D:\Conda\envs\env_isaaclab\python.exe -B .\quad_leg_xu\quad_leg_trot\quad_leg_trot1.py --keyboard
# python .\quad_leg_xu\quad_leg_trot\quad_leg_trot1.py --keyboard --imu_port COM4
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from bennett_deploy import contract as contract_module
from bennett_deploy import policy as policy_module
from bennett_deploy import runtime
from bennett_deploy.gait import GaitTerms


DEFAULT_POLICY = (
    r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_trot\quad_leg_trot1"
    r"\flat\2026-07-27_20-08-08\exported\policy.pt"
)

EXPECTED_OBSERVATION_ORDER = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
    "trot_phase",
    "trot_leg_phase",
    "desired_contacts",
    "gait_params",
)

EXPECTED_OBSERVATION_FUNCTIONS = {
    "observations.policy.trot_phase.func": (
        "bennett_rl.tasks.manager_based.quad_leg_trot.quad_leg_trot1.mdp.gait:"
        "commanded_trot_global_phase_sin_cos"
    ),
    "observations.policy.trot_leg_phase.func": (
        "bennett_rl.tasks.manager_based.quad_leg_trot.quad_leg_trot1.mdp.gait:"
        "commanded_trot_leg_phase_sin_cos"
    ),
    "observations.policy.desired_contacts.func": (
        "bennett_rl.tasks.manager_based.quad_leg_trot.quad_leg_trot1.mdp.gait:"
        "commanded_trot_desired_contacts"
    ),
    "observations.policy.gait_params.func": (
        "bennett_rl.tasks.manager_based.quad_leg_trot.quad_leg_trot1.mdp.gait:"
        "commanded_trot_gait_params"
    ),
}

EXPECTED_TARGET_LIMITS = {
    "actions.joint_pos.clip..*_thigh": (-0.80, 0.80),
    "actions.joint_pos.clip..*_calf": (-0.90, 0.55),
}


def _require(nodes, key, expected_type=None):
    return contract_module._required(nodes, key, expected_type)


def _consistent(nodes, prefix, leaf):
    return contract_module._consistent_under(nodes, prefix, leaf)


@dataclass(frozen=True)
class Trot1DeploymentContract(contract_module.DeploymentContract):
    min_frequency_hz: float
    max_frequency_hz: float
    min_equivalent_speed: float
    max_equivalent_speed: float
    low_speed_duty_factor: float
    high_speed_duty_factor: float
    yaw_equivalent_radius: float

    @classmethod
    def load(cls, policy_arg, *, action_clip_override=None):
        policy_path = contract_module.resolve_policy_path(policy_arg)
        if not policy_path.is_file():
            raise FileNotFoundError(f"exported policy not found: {policy_path}")
        run_dir = policy_path.parent.parent if policy_path.parent.name == "exported" else policy_path.parent
        env_yaml = run_dir / "params" / "env.yaml"
        agent_yaml = run_dir / "params" / "agent.yaml"
        for required_path in (env_yaml, agent_yaml):
            if not required_path.is_file():
                raise FileNotFoundError(f"refusing hardware deployment without exact run config: {required_path}")

        env = contract_module.parse_isaaclab_yaml(env_yaml)
        agent = contract_module.parse_isaaclab_yaml(agent_yaml)
        sim_dt = float(_require(env, "sim.dt", (int, float)))
        decimation = int(_require(env, "decimation", (int, float)))
        if sim_dt <= 0.0 or decimation <= 0:
            raise ValueError("sim.dt and decimation must be positive")

        action_joint_names = tuple(_require(env, "actions.joint_pos.joint_names", list))
        if action_joint_names != contract_module.JOINT_ORDER:
            raise ValueError(
                f"action joint order mismatch: trained={action_joint_names}, "
                f"deploy={contract_module.JOINT_ORDER}"
            )
        if _require(env, "actions.joint_pos.use_default_offset", bool) is not True:
            raise ValueError("Trot1 deployment requires actions.joint_pos.use_default_offset=true")
        if _require(env, "actions.joint_pos.preserve_order", bool) is not True:
            raise ValueError("Trot1 deployment requires actions.joint_pos.preserve_order=true")
        if float(_require(env, "actions.joint_pos.offset", (int, float))) != 0.0:
            raise ValueError("Trot1 deployment requires actions.joint_pos.offset=0")
        expected_action_class = "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction"
        if _require(env, "actions.joint_pos.class_type", str) != expected_action_class:
            raise ValueError("Trot1 deployment only supports JointPositionAction")

        observation_order = tuple(
            name
            for name in _require(env, "@order:observations.policy", list)
            if env.get(f"observations.policy.{name}.func") is not None
        )
        if observation_order != EXPECTED_OBSERVATION_ORDER:
            raise ValueError(
                f"policy observation order mismatch: trained={observation_order}, "
                f"deploy={EXPECTED_OBSERVATION_ORDER}"
            )
        for term, function in (
            ("joint_pos", "isaaclab.envs.mdp.observations:joint_pos_rel"),
            ("joint_vel", "isaaclab.envs.mdp.observations:joint_vel_rel"),
        ):
            prefix = f"observations.policy.{term}"
            if _require(env, prefix + ".func", str) != function:
                raise ValueError(f"unsupported {term} observation function")
            observed_names = tuple(_require(env, prefix + ".params.asset_cfg.joint_names", list))
            if observed_names != contract_module.JOINT_ORDER:
                raise ValueError(f"{term} observation joint order mismatch: {observed_names}")
            if _require(env, prefix + ".params.asset_cfg.preserve_order", bool) is not True:
                raise ValueError(f"{term} observation must preserve joint order")

        for key, expected in EXPECTED_OBSERVATION_FUNCTIONS.items():
            actual = env.get(key)
            if actual != expected:
                raise ValueError(f"Trot1 deployment requires {key}={expected!r}, got {actual!r}")
        for key, expected in (
            ("commands.base_velocity.min_abs_lin_vel_x", 0.08),
            ("commands.base_velocity.min_abs_ang_vel_z", 0.18),
        ):
            actual = env.get(key)
            if actual is None or not math.isclose(float(actual), expected, abs_tol=1.0e-9):
                raise ValueError(f"Trot1 deployment requires {key}={expected}, got {actual!r}")
        lateral_range = env.get("commands.base_velocity.ranges.lin_vel_y")
        if lateral_range != [0.0, 0.0]:
            raise ValueError(f"Trot1 was not trained for lateral commands, got lin_vel_y={lateral_range!r}")
        for key, expected in EXPECTED_TARGET_LIMITS.items():
            actual = env.get(key)
            if not isinstance(actual, list) or len(actual) != 2:
                raise ValueError(f"Trot1 requires two target limits at {key}, got {actual!r}")
            if not all(
                math.isclose(float(value), limit, rel_tol=0.0, abs_tol=1.0e-9)
                for value, limit in zip(actual, expected)
            ):
                raise ValueError(f"Trot1 requires {key}={expected!r}, got {actual!r}")

        defaults = tuple(
            float(_require(env, f"scene.robot.init_state.joint_pos.{name}", (int, float)))
            for name in contract_module.JOINT_ORDER
        )
        action_scale = float(_require(env, "actions.joint_pos.scale", (int, float)))
        if action_scale <= 0.0:
            raise ValueError("actions.joint_pos.scale must be positive")
        trained_clip = agent.get("clip_actions")
        if action_clip_override is not None:
            action_clip = float(action_clip_override)
            clip_source = "cli_override"
        elif trained_clip is None:
            action_clip = contract_module.UNCLIPPED_RUNNER_DEPLOYMENT_CLIP
            clip_source = "explicit_deployment_guard_for_unclipped_runner"
        else:
            action_clip = float(trained_clip)
            clip_source = "agent.yaml"
        if not math.isfinite(action_clip) or action_clip <= 0.0:
            raise ValueError("deployment action clip must be finite and positive")

        gait_prefix = "observations.policy."
        min_frequency = _consistent(env, gait_prefix, "min_frequency_hz")
        max_frequency = _consistent(env, gait_prefix, "max_frequency_hz")
        min_speed = _consistent(env, gait_prefix, "min_equivalent_speed")
        max_speed = _consistent(env, gait_prefix, "max_equivalent_speed")
        low_duty = _consistent(env, gait_prefix, "low_speed_duty_factor")
        high_duty = _consistent(env, gait_prefix, "high_speed_duty_factor")
        swing_height = _consistent(env, gait_prefix, "swing_height")
        yaw_radius = _consistent(env, gait_prefix, "yaw_equivalent_radius")
        command_deadband = _consistent(env, gait_prefix, "command_deadband")

        policy_sha256 = contract_module._sha256(policy_path)
        env_sha256 = contract_module._sha256(env_yaml)
        agent_sha256 = contract_module._sha256(agent_yaml)
        fingerprint_values = {
            "policy_sha256": policy_sha256,
            "env_sha256": env_sha256,
            "agent_sha256": agent_sha256,
            "joint_order": contract_module.JOINT_ORDER,
            "joint_specs": tuple(asdict(spec) for spec in contract_module.JOINT_SPECS),
            "train_default_rad": defaults,
            "sim_dt": sim_dt,
            "decimation": decimation,
            "action_scale_rad": action_scale,
            "action_clip": action_clip,
            "min_frequency_hz": min_frequency,
            "max_frequency_hz": max_frequency,
            "min_equivalent_speed": min_speed,
            "max_equivalent_speed": max_speed,
            "low_speed_duty_factor": low_duty,
            "high_speed_duty_factor": high_duty,
            "swing_height": swing_height,
            "yaw_equivalent_radius": yaw_radius,
            "command_deadband": command_deadband,
        }
        fingerprint_payload = json.dumps(
            fingerprint_values, sort_keys=True, separators=(",", ":"), default=list
        ).encode("utf-8")
        fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()[:16]

        return cls(
            policy_path=policy_path,
            run_dir=run_dir,
            env_yaml=env_yaml,
            agent_yaml=agent_yaml,
            policy_sha256=policy_sha256,
            env_sha256=env_sha256,
            agent_sha256=agent_sha256,
            joint_order=contract_module.JOINT_ORDER,
            joint_specs=contract_module.JOINT_SPECS,
            train_default_rad=defaults,
            observation_dim=contract_module.OBSERVATION_DIM,
            action_dim=contract_module.ACTION_DIM,
            sim_dt=sim_dt,
            decimation=decimation,
            policy_rate_hz=1.0 / (sim_dt * decimation),
            action_scale_rad=action_scale,
            action_clip=action_clip,
            action_clip_source=clip_source,
            stiffness=_consistent(env, "scene.robot.actuators.", "stiffness"),
            damping=_consistent(env, "scene.robot.actuators.", "damping"),
            effort_limit=_consistent(env, "scene.robot.actuators.", "effort_limit"),
            saturation_effort=_consistent(env, "scene.robot.actuators.", "saturation_effort"),
            velocity_limit=_consistent(env, "scene.robot.actuators.", "velocity_limit"),
            gait_frequency_hz=min_frequency,
            gait_duty_factor=low_duty,
            gait_swing_height=swing_height,
            command_deadband=command_deadband,
            fingerprint=fingerprint,
            min_frequency_hz=min_frequency,
            max_frequency_hz=max_frequency,
            min_equivalent_speed=min_speed,
            max_equivalent_speed=max_speed,
            low_speed_duty_factor=low_duty,
            high_speed_duty_factor=high_duty,
            yaw_equivalent_radius=yaw_radius,
        )

    def summary_lines(self):
        for line in super().summary_lines():
            if line.startswith("[CONTRACT] gait frequency="):
                yield (
                    f"[CONTRACT] trot frequency={self.min_frequency_hz:.3f}.."
                    f"{self.max_frequency_hz:.3f}Hz duty={self.low_speed_duty_factor:.3f}.."
                    f"{self.high_speed_duty_factor:.3f} swing_height="
                    f"{self.gait_swing_height:.3f}m deadband={self.command_deadband:.3f}"
                )
            else:
                yield line


class Trot1Clock:
    """Mirror the stateful speed-conditioned clock in Trot1 training."""

    OFFSETS = (0.0, 0.5, 0.5, 0.0)

    def __init__(self, contract: Trot1DeploymentContract):
        self.contract = contract
        self.dt = 1.0 / contract.policy_rate_hz
        self.phase = 0.0
        self.was_moving = False

    def reset(self) -> None:
        self.phase = 0.0
        self.was_moving = False

    def _parameters(self, command: tuple[float, float, float]):
        moving = math.sqrt(sum(value * value for value in command)) >= self.contract.command_deadband
        if not moving:
            return 0.0, 1.0, 0.0, False
        equivalent_speed = math.hypot(command[0], command[1])
        equivalent_speed += self.contract.yaw_equivalent_radius * abs(command[2])
        blend = (
            (equivalent_speed - self.contract.min_equivalent_speed)
            / (self.contract.max_equivalent_speed - self.contract.min_equivalent_speed)
        )
        blend = max(0.0, min(1.0, blend))
        frequency = self.contract.min_frequency_hz + blend * (
            self.contract.max_frequency_hz - self.contract.min_frequency_hz
        )
        duty = self.contract.low_speed_duty_factor + blend * (
            self.contract.high_speed_duty_factor - self.contract.low_speed_duty_factor
        )
        return frequency, duty, self.contract.gait_swing_height, True

    def update(self, command: Sequence[float]) -> GaitTerms:
        if len(command) != 3:
            raise ValueError("velocity command must contain vx, vy, yaw")
        command_tuple = tuple(float(value) for value in command)
        frequency, duty, height, moving = self._parameters(command_tuple)
        if not moving:
            self.phase = 0.0
            self.was_moving = False
            return GaitTerms(
                velocity_command=(0.0, 0.0, 0.0),
                global_phase=(0.0, 1.0),
                leg_phase=(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
                desired_contacts=(1.0, 1.0, 1.0, 1.0),
                gait_params=(0.0, 1.0, 0.0),
                phase=0.0,
            )

        if not self.was_moving:
            # Training starts at the end of the diagonal swing interval.
            self.phase = 1.0 - duty
        else:
            self.phase = (self.phase + self.dt * frequency) % 1.0
        self.was_moving = True

        leg_values = tuple((self.phase - offset) % 1.0 for offset in self.OFFSETS)
        swing_fraction = 1.0 - duty
        desired_contacts = tuple(0.0 if value < swing_fraction else 1.0 for value in leg_values)
        leg_sines = tuple(math.sin(2.0 * math.pi * value) for value in leg_values)
        leg_cosines = tuple(math.cos(2.0 * math.pi * value) for value in leg_values)
        phase_angle = 2.0 * math.pi * self.phase
        return GaitTerms(
            velocity_command=command_tuple,
            global_phase=(math.sin(phase_angle), math.cos(phase_angle)),
            # Trot1 training concatenates all four sines, then all four cosines.
            leg_phase=leg_sines + leg_cosines,
            desired_contacts=desired_contacts,
            gait_params=(frequency, duty, height),
            phase=self.phase,
        )


class Trot1ObservationBuilder:
    def __init__(self, contract: Trot1DeploymentContract, torch_module):
        self.contract = contract
        self.torch = torch_module
        self.clock = Trot1Clock(contract)

    def reset(self) -> None:
        self.clock.reset()

    def build(
        self,
        *,
        base_ang_vel: Sequence[float],
        projected_gravity: Sequence[float],
        velocity_command: Sequence[float],
        joint_pos_rel: Sequence[float],
        joint_vel_rel: Sequence[float],
        last_action,
        phase_elapsed_s: float,
    ):
        del phase_elapsed_s
        gait = self.clock.update(velocity_command)
        values = (
            list(base_ang_vel)
            + list(projected_gravity)
            + list(gait.velocity_command)
            + list(joint_pos_rel)
            + list(joint_vel_rel)
            + list(last_action.tolist())
            + list(gait.global_phase)
            + list(gait.leg_phase)
            + list(gait.desired_contacts)
            + list(gait.gait_params)
        )
        if len(values) != self.contract.observation_dim:
            raise RuntimeError(
                f"observation length mismatch: {len(values)} != {self.contract.observation_dim}"
            )
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("observation contains non-finite values")
        observation = self.torch.tensor(values, dtype=self.torch.float32).unsqueeze(0)
        return observation, gait


class Trot1PolicyPipeline(policy_module.PolicyPipeline):
    def __init__(self, contract, *, output_scale: float, target_rate_limit_deg_s: float):
        super().__init__(
            contract,
            output_scale=output_scale,
            target_rate_limit_deg_s=target_rate_limit_deg_s,
        )
        self.observations = Trot1ObservationBuilder(contract, self.torch)
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
        target = self.torch.maximum(
            self.target_lower,
            self.torch.minimum(self.target_upper, step.target_sim_rad),
        )
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


class Trot1DeploymentRunner(runtime.DeploymentRunner):
    """Print the Trot1 observation contract without crawl-specific labels."""

    def print_contract(self) -> None:
        for line in self.contract.summary_lines():
            print(line)
        print(
            "[OBS] base_ang_vel(3), projected_gravity(3), velocity_command(3), "
            "joint_pos_rel(8), joint_vel_rel(8), last_action(8), trot_phase(2), "
            "trot_leg_phase(8), desired_contacts(4), gait_params(3)"
        )
        print("[JOINT-ORDER] " + ", ".join(self.contract.joint_order))
        print(
            "[TRAIN-DEFAULT] "
            + ", ".join(
                f"{name}:{math.degrees(self.contract.train_default_rad[index]):+.1f}deg"
                for index, name in enumerate(self.contract.joint_order)
            )
        )
        print(
            f"[RUNTIME] kp={self.kp:.3f} kd={self.kd:.3f} "
            f"output_scale={self.args.output_scale:.3f} active={self.args.active_joints} "
            f"control_rate={self.args.control_rate_hz:.1f}Hz "
            f"tau_stop={self.args.tau_limit:.3f}Nm"
        )
        if self.dm_sdk_dir is not None:
            print(
                f"[DM-SDK] profile={self.args.dm_sdk_profile} path={self.dm_sdk_dir} "
                f"libdm_sha256={self.dm_sdk_sha256}"
            )
        print(
            "[ACTION] q_cmd = motor_zero + sim_to_motor * "
            "clip(train_default + rate_limited_policy_offset, trained_joint_limits)"
        )


def _add_trot1_defaults(argv: list[str]) -> None:
    defaults = {
        "--keyboard_step_x": "0.02",
        "--keyboard_min_nonzero_x": "0.08",
        "--keyboard_min_x": "-0.35",
        "--keyboard_max_x": "0.35",
        "--keyboard_step_y": "0.01",
        "--keyboard_min_y": "0.0",
        "--keyboard_max_y": "0.0",
        "--keyboard_step_yaw": "0.18",
        "--keyboard_min_yaw": "-0.60",
        "--keyboard_max_yaw": "0.60",
        "--control_rate_hz": "250.0",
        "--dm_sdk_profile": "low_latency",
        "--motor_feedback_timeout_s": "0.10",
        "--default_mode": "motor_zero",
        # Offline phase sweeps require about 330-375 deg/s at the policy p95.
        # 360 deg/s preserves nominal trot and most high-speed motion while
        # still rejecting the policy's rare 1000+ deg/s discontinuities.
        "--target_rate_limit_deg_s": "360.0",
    }
    present = {item.split("=", 1)[0] for item in argv[1:] if item.startswith("--")}
    for flag, value in defaults.items():
        if flag not in present:
            argv.extend((flag, value))


def main() -> int:
    _add_trot1_defaults(sys.argv)
    runtime.DEFAULT_POLICY = DEFAULT_POLICY
    runtime.DeploymentContract = Trot1DeploymentContract
    runtime.PolicyPipeline = Trot1PolicyPipeline
    runtime.DeploymentRunner = Trot1DeploymentRunner
    return runtime.main()


if __name__ == "__main__":
    raise SystemExit(main())
