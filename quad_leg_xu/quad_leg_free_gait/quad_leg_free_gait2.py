"""Bennett FreeGait2 hardware deployment entry point.

The policy has no prescribed gait clock.  Its exact 33-value observation is:
base angular velocity, projected gravity, velocity command, relative joint
position, relative joint velocity, and the previous policy action.

``--check_only`` validates the saved policy/config contract without opening
the IMU, USB-CAN adapter, or motors.
"""

# Run from E:\HuanCun\Desktop\u2canfd:
# D:\Conda\envs\env_isaaclab\python.exe -B .\quad_leg_xu\quad_leg_free_gait\quad_leg_free_gait2.py --check_only
# D:\Conda\envs\env_isaaclab\python.exe -B .\quad_leg_xu\quad_leg_free_gait\quad_leg_free_gait2.py --keyboard --log_csv ".\quad_leg_xu\quad_leg_free_gait\logs\free-gait2-first.csv"

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict
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
    r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_free_gait"
    r"\quad_leg_free_gait2\flat\2026-07-29_04-57-26\exported\policy.pt"
)

OBSERVATION_DIM = 33
EXPECTED_OBSERVATION_ORDER = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
)
EXPECTED_OBSERVATION_FUNCTIONS = {
    "base_ang_vel": "isaaclab.envs.mdp.observations:base_ang_vel",
    "projected_gravity": "isaaclab.envs.mdp.observations:projected_gravity",
    "velocity_commands": "isaaclab.envs.mdp.observations:generated_commands",
    "joint_pos": "isaaclab.envs.mdp.observations:joint_pos_rel",
    "joint_vel": "isaaclab.envs.mdp.observations:joint_vel_rel",
    "actions": "isaaclab.envs.mdp.observations:last_action",
}
EXPECTED_COMMAND_MINIMA = {
    "commands.base_velocity.min_abs_lin_vel_x": 0.08,
    "commands.base_velocity.min_abs_lin_vel_y": 0.06,
    "commands.base_velocity.min_abs_ang_vel_z": 0.18,
}
EXPECTED_COMMAND_RANGES = {
    "commands.base_velocity.ranges.lin_vel_x": (-0.35, 0.35),
    "commands.base_velocity.ranges.lin_vel_y": (-0.20, 0.20),
    "commands.base_velocity.ranges.ang_vel_z": (-0.60, 0.60),
}
EXPECTED_TARGET_LIMITS = {
    "actions.joint_pos.clip..*_thigh": (-0.80, 0.80),
    "actions.joint_pos.clip..*_calf": (-0.90, 0.55),
}
EXPECTED_MOTOR = {
    "effort_limit": 8.0,
    "saturation_effort": 20.0,
    "velocity_limit": 19.896753,
}
EXPECTED_FREE_GAIT_REWARD = (
    "bennett_rl.tasks.manager_based.quad_leg_free_gait.quad_leg_free_gait2."
    "mdp.rewards:gait_free_leg_lift_starvation_l2"
)


def _require(nodes, key, expected_type=None):
    return contract_module._required(nodes, key, expected_type)


def _consistent(nodes, prefix, leaf):
    return contract_module._consistent_under(nodes, prefix, leaf)


def _require_close(nodes, key: str, expected: float) -> None:
    actual = _require(nodes, key, (int, float))
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"FreeGait2 requires {key}={expected}, got {actual!r}")


def _require_pair(nodes, key: str, expected: tuple[float, float]) -> None:
    actual = _require(nodes, key, list)
    if len(actual) != 2 or any(
        not math.isclose(float(value), limit, rel_tol=0.0, abs_tol=1.0e-9)
        for value, limit in zip(actual, expected)
    ):
        raise ValueError(f"FreeGait2 requires {key}={expected!r}, got {actual!r}")


class FreeGait2DeploymentContract(contract_module.DeploymentContract):
    """Reject any policy whose saved config is not the exact FreeGait2 run."""

    @classmethod
    def load(cls, policy_arg, *, action_clip_override=None):
        policy_path = contract_module.resolve_policy_path(policy_arg)
        if not policy_path.is_file():
            raise FileNotFoundError(f"exported policy not found: {policy_path}")
        run_dir = (
            policy_path.parent.parent
            if policy_path.parent.name == "exported"
            else policy_path.parent
        )
        env_yaml = run_dir / "params" / "env.yaml"
        agent_yaml = run_dir / "params" / "agent.yaml"
        for required_path in (env_yaml, agent_yaml):
            if not required_path.is_file():
                raise FileNotFoundError(
                    f"refusing hardware deployment without exact run config: {required_path}"
                )

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
            raise ValueError("FreeGait2 requires use_default_offset=true")
        if _require(env, "actions.joint_pos.preserve_order", bool) is not True:
            raise ValueError("FreeGait2 requires preserve_order=true")
        if float(_require(env, "actions.joint_pos.offset", (int, float))) != 0.0:
            raise ValueError("FreeGait2 requires actions.joint_pos.offset=0")
        if (
            _require(env, "actions.joint_pos.class_type", str)
            != "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction"
        ):
            raise ValueError("FreeGait2 deployment only supports JointPositionAction")

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
        for term, expected_function in EXPECTED_OBSERVATION_FUNCTIONS.items():
            actual_function = _require(env, f"observations.policy.{term}.func", str)
            if actual_function != expected_function:
                raise ValueError(
                    f"FreeGait2 requires observations.policy.{term}.func="
                    f"{expected_function!r}, got {actual_function!r}"
                )
        for term in ("joint_pos", "joint_vel"):
            prefix = f"observations.policy.{term}"
            observed_names = tuple(
                _require(env, prefix + ".params.asset_cfg.joint_names", list)
            )
            if observed_names != contract_module.JOINT_ORDER:
                raise ValueError(f"{term} observation joint order mismatch: {observed_names}")
            if _require(env, prefix + ".params.asset_cfg.preserve_order", bool) is not True:
                raise ValueError(f"{term} observation must preserve joint order")

        reward_function = _require(env, "rewards.leg_lift_starvation.func", str)
        if reward_function != EXPECTED_FREE_GAIT_REWARD:
            raise ValueError(
                "policy is not the four-leg-participation FreeGait2 task: "
                f"{reward_function!r}"
            )
        for key, expected in EXPECTED_COMMAND_MINIMA.items():
            _require_close(env, key, expected)
        for key, expected in EXPECTED_COMMAND_RANGES.items():
            _require_pair(env, key, expected)
        for key, expected in EXPECTED_TARGET_LIMITS.items():
            _require_pair(env, key, expected)

        defaults = tuple(
            float(
                _require(
                    env,
                    f"scene.robot.init_state.joint_pos.{name}",
                    (int, float),
                )
            )
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

        motor_values = {
            key: _consistent(env, "scene.robot.actuators.", key)
            for key in EXPECTED_MOTOR
        }
        for key, expected in EXPECTED_MOTOR.items():
            if not math.isclose(
                motor_values[key], expected, rel_tol=0.0, abs_tol=1.0e-6
            ):
                raise ValueError(
                    f"FreeGait2 requires motor {key}={expected}, "
                    f"got {motor_values[key]}"
                )

        command_deadband = float(
            _require(
                env,
                "rewards.leg_lift_starvation.params.command_deadband",
                (int, float),
            )
        )
        policy_sha256 = contract_module._sha256(policy_path)
        env_sha256 = contract_module._sha256(env_yaml)
        agent_sha256 = contract_module._sha256(agent_yaml)
        stiffness = _consistent(env, "scene.robot.actuators.", "stiffness")
        damping = _consistent(env, "scene.robot.actuators.", "damping")
        fingerprint_values = {
            "task": "quad_leg_free_gait2",
            "policy_sha256": policy_sha256,
            "env_sha256": env_sha256,
            "agent_sha256": agent_sha256,
            "joint_order": contract_module.JOINT_ORDER,
            "joint_specs": tuple(
                asdict(spec) for spec in contract_module.JOINT_SPECS
            ),
            "train_default_rad": defaults,
            "observation_dim": OBSERVATION_DIM,
            "sim_dt": sim_dt,
            "decimation": decimation,
            "action_scale_rad": action_scale,
            "action_clip": action_clip,
            "stiffness": stiffness,
            "damping": damping,
            "motor": motor_values,
            "command_deadband": command_deadband,
        }
        fingerprint_payload = json.dumps(
            fingerprint_values,
            sort_keys=True,
            separators=(",", ":"),
            default=list,
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
            observation_dim=OBSERVATION_DIM,
            action_dim=contract_module.ACTION_DIM,
            sim_dt=sim_dt,
            decimation=decimation,
            policy_rate_hz=1.0 / (sim_dt * decimation),
            action_scale_rad=action_scale,
            action_clip=action_clip,
            action_clip_source=clip_source,
            stiffness=stiffness,
            damping=damping,
            effort_limit=motor_values["effort_limit"],
            saturation_effort=motor_values["saturation_effort"],
            velocity_limit=motor_values["velocity_limit"],
            # These legacy fields are not policy observations for FreeGait2.
            # A finite value keeps the shared synthetic preflight length valid.
            gait_frequency_hz=1.0,
            gait_duty_factor=1.0,
            gait_swing_height=0.0,
            command_deadband=command_deadband,
            fingerprint=fingerprint,
        )

    def summary_lines(self):
        yield f"[CONTRACT] id={self.fingerprint} policy={self.policy_path}"
        yield f"[CONTRACT] policy_sha256={self.policy_sha256}"
        yield f"[CONTRACT] env={self.env_yaml} sha256={self.env_sha256}"
        yield f"[CONTRACT] agent={self.agent_yaml} sha256={self.agent_sha256}"
        yield (
            f"[CONTRACT] rate={self.policy_rate_hz:.3f}Hz "
            f"action_scale={self.action_scale_rad:.6f}rad "
            f"action_clip=+/-{self.action_clip:.3f}({self.action_clip_source}) "
            f"kp={self.stiffness:.3f} kd={self.damping:.3f}"
        )
        yield "[CONTRACT] gait=free; no phase, foot order, or desired-contact input"
        yield (
            f"[CONTRACT] actuator effort={self.effort_limit:.3f}Nm "
            f"saturation={self.saturation_effort:.3f}Nm "
            f"velocity={self.velocity_limit:.6f}rad/s"
        )


class FreeGait2ObservationBuilder:
    def __init__(self, contract: FreeGait2DeploymentContract, torch_module):
        self.contract = contract
        self.torch = torch_module

    def reset(self) -> None:
        return None

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
        command = tuple(float(value) for value in velocity_command)
        values = (
            list(base_ang_vel)
            + list(projected_gravity)
            + list(command)
            + list(joint_pos_rel)
            + list(joint_vel_rel)
            + list(last_action.tolist())
        )
        if len(values) != self.contract.observation_dim:
            raise RuntimeError(
                f"observation length mismatch: {len(values)} != "
                f"{self.contract.observation_dim}"
            )
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("observation contains non-finite values")
        observation = self.torch.tensor(
            values, dtype=self.torch.float32
        ).unsqueeze(0)
        gait = GaitTerms(
            velocity_command=command,
            global_phase=(0.0, 1.0),
            leg_phase=(0.0, 1.0) * 4,
            desired_contacts=(1.0, 1.0, 1.0, 1.0),
            gait_params=(0.0, 0.0, 0.0),
            phase=0.0,
        )
        return observation, gait


class FreeGait2PolicyPipeline(policy_module.PolicyPipeline):
    """Use the 33-value policy input and the trained absolute target limits."""

    def __init__(self, contract, *, output_scale: float, target_rate_limit_deg_s: float):
        super().__init__(
            contract,
            output_scale=output_scale,
            target_rate_limit_deg_s=target_rate_limit_deg_s,
        )
        self.observations = FreeGait2ObservationBuilder(contract, self.torch)
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


class FreeGait2DeploymentRunner(runtime.DeploymentRunner):
    def print_contract(self) -> None:
        for line in self.contract.summary_lines():
            print(line)
        print(
            "[OBS] base_ang_vel(3), projected_gravity(3), velocity_command(3), "
            "joint_pos_rel(8), joint_vel_rel(8), last_action(8)"
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
            f"output_scale={self.args.output_scale:.3f} "
            f"active={self.args.active_joints} "
            f"control_rate={self.args.control_rate_hz:.1f}Hz "
            f"effort_envelope={self.args.software_effort_limit_nm:.3f}Nm "
            f"tau_stop={self.args.tau_limit:.3f}Nm"
        )
        if self.dm_sdk_dir is not None:
            print(
                f"[DM-SDK] profile={self.args.dm_sdk_profile} "
                f"path={self.dm_sdk_dir} libdm_sha256={self.dm_sdk_sha256}"
            )
        print(
            "[ACTION] q_cmd = motor_zero + sim_to_motor * "
            "clip(train_default + rate_limited_policy_offset, trained_joint_limits), "
            "then live-feedback MIT-PD effort limiting"
        )


def _add_free_gait2_defaults(argv: list[str]) -> None:
    defaults = {
        "--keyboard_step_x": "0.02",
        "--keyboard_min_nonzero_x": "0.08",
        "--keyboard_min_x": "-0.35",
        "--keyboard_max_x": "0.35",
        "--keyboard_step_y": "0.06",
        "--keyboard_min_y": "-0.20",
        "--keyboard_max_y": "0.20",
        "--keyboard_step_yaw": "0.18",
        "--keyboard_min_yaw": "-0.60",
        "--keyboard_max_yaw": "0.60",
        "--control_rate_hz": "250.0",
        "--dm_sdk_profile": "low_latency",
        "--motor_feedback_timeout_s": "0.10",
        "--default_mode": "motor_zero",
        # Preserve the trained residual amplitude; safety matching is handled
        # by the trained actuator effort envelope below.
        "--output_scale": "1",
        # Offline 50 Hz diagnostics measured a 322 deg/s policy-step p95 in
        # the most abrupt combined command.  This guard preserves that bulk
        # behavior while rejecting rarer discontinuities.
        "--target_rate_limit_deg_s": "360.0",
        # Reproduce the training DCMotor's continuous 8 Nm cap.  The motor's
        # 20 Nm saturation_effort remains the torque-speed envelope peak, not
        # the continuous policy effort used by Isaac Lab.
        "--software_effort_limit_nm": "8.0",
    }
    present = {
        item.split("=", 1)[0]
        for item in argv[1:]
        if item.startswith("--")
    }
    for flag, value in defaults.items():
        if flag not in present:
            argv.extend((flag, value))


def main() -> int:
    _add_free_gait2_defaults(sys.argv)
    runtime.DEFAULT_POLICY = DEFAULT_POLICY
    runtime.DeploymentContract = FreeGait2DeploymentContract
    runtime.PolicyPipeline = FreeGait2PolicyPipeline
    runtime.DeploymentRunner = FreeGait2DeploymentRunner
    return runtime.main()


if __name__ == "__main__":
    raise SystemExit(main())
