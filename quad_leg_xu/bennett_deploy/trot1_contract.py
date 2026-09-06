"""Trot1-specific deployment contract.

This is the document-adapter for the `quad_leg_trot1` policy. Unlike the
go2-10 contract, trot1's gait is SPEED-CONDITIONED: frequency and stance duty
factor interpolate with the commanded velocity, and the phase is a *stateful
integrator* (phase += dt * f(cmd)), not a pure function of wall-clock time.
So this contract carries the full gait config (a TrotGaitConfig) instead of
fixed `gait_frequency_hz` / `gait_duty_factor` scalars.

Hard-won, POLICY-INDEPENDENT facts baked in here (they survive any re-train
of Bennett):
  * joint sign table (sim_to_motor) = the go2/Bennett layout sign table, with
    FL/RL_thigh = +1. The -1 (taken from 3_quad_leg_track.py) was tried first
    and produced a MIRRORED GAIT on the real robot, so it is wrong for trot1.
    sign is applied symmetrically (feedback /sign, command *sign, motor_zero)
    in runtime.py, so flipping an axis is always control-stable.
  * the observation order (below) and the grouped leg-phase layout are matched
    EXACTLY to how `commanded_trot_leg_phase_sin_cos` concatenates in the sim.
  * the diagonal phase offsets (FL+RR / FR+RL) and the stateful clock are
    replicated in trot1_gait.py, isolated so a future gaited/ungated redesign
    swaps only that module.

The parts that DO change if the gait is redesigned (e.g. to an unconstrained
RL gait) are the four gait observation terms below + the gait config; the
obs assembly, sign table, action decode and actuator gains carry over.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .contract import (
    JOINT_ORDER,
    JointSpec,
    _consistent_under,
    _required,
    _sha256,
    parse_isaaclab_yaml,
    resolve_policy_path,
)

OBSERVATION_DIM = 50          # 3+3+3+8+8+8 + 2 + 8 + 4 + 3
ACTION_DIM = 8
UNCLIPPED_RUNNER_DEPLOYMENT_CLIP = 3.5

# Sign table (sim_to_motor). FL/RL_thigh = +1 matches the go2/Bennett layout.
# The -1 from 3_quad_leg_track.py made trot1 mirror its gait on the real
# robot, so trot1 uses +1 here (verified on hardware 2026-09-05).
JOINT_SPECS = (
    JointSpec("FL_thigh", 0, 0x01, 0x11, +1.0),
    JointSpec("FL_calf", 0, 0x02, 0x12, +1.0),
    JointSpec("FR_thigh", 1, 0x03, 0x13, +1.0),
    JointSpec("FR_calf", 1, 0x04, 0x14, -1.0),
    JointSpec("RL_thigh", 0, 0x05, 0x15, +1.0),
    JointSpec("RL_calf", 0, 0x06, 0x16, +1.0),
    JointSpec("RR_thigh", 1, 0x07, 0x17, +1.0),
    JointSpec("RR_calf", 1, 0x08, 0x18, -1.0),
)

# Order of observation terms as concatenated in the trot1 sim (env.yaml
# `@order:observations.policy`, null terms dropped). Each must be built by the
# deployment observation builder in exactly this position to reproduce the
# tensor the policy was trained on.
EXPECTED_OBS_ORDER = (
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

# The four gait terms all repeat the same param block; we require them equal.
GAIT_TERMS = ("trot_phase", "trot_leg_phase", "desired_contacts", "gait_params")
GAIT_LEAFS = (
    "command_deadband",
    "min_frequency_hz",
    "max_frequency_hz",
    "min_equivalent_speed",
    "max_equivalent_speed",
    "low_speed_duty_factor",
    "high_speed_duty_factor",
    "swing_height",
    "yaw_equivalent_radius",
)


@dataclass(frozen=True)
class TrotGaitConfig:
    """Speed-conditioned gait parameters read from the trot1 env.yaml."""

    command_deadband: float
    min_frequency_hz: float
    max_frequency_hz: float
    min_equivalent_speed: float
    max_equivalent_speed: float
    low_speed_duty_factor: float
    high_speed_duty_factor: float
    swing_height: float
    yaw_equivalent_radius: float


def _gate_value(env: dict, leaf: str) -> float:
    """Read one gait param, requiring it identical across all four gait terms."""
    values = [
        float(env[f"observations.policy.{term}.params.{leaf}"])
        for term in GAIT_TERMS
        if f"observations.policy.{term}.params.{leaf}" in env
    ]
    if len(values) != len(GAIT_TERMS):
        raise ValueError(f"trot1 gait term missing '{leaf}' in one or more terms: {len(values)}/{len(GAIT_TERMS)}")
    first = values[0]
    if any(not math.isclose(value, first, rel_tol=0.0, abs_tol=1e-9) for value in values[1:]):
        raise ValueError(f"trot1 gait '{leaf}' inconsistent across gait terms: {values}")
    return first


@dataclass(frozen=True)
class TrotDeploymentContract:
    policy_path: Path
    run_dir: Path
    env_yaml: Path
    agent_yaml: Path
    policy_sha256: str
    env_sha256: str
    agent_sha256: str
    joint_order: tuple[str, ...]
    joint_specs: tuple[JointSpec, ...]
    train_default_rad: tuple[float, ...]
    observation_dim: int
    action_dim: int
    sim_dt: float
    decimation: int
    policy_rate_hz: float
    action_scale_rad: float
    action_clip: float
    action_clip_source: str
    stiffness: float
    damping: float
    effort_limit: float
    saturation_effort: float
    velocity_limit: float
    gait: TrotGaitConfig
    fingerprint: str

    @classmethod
    def load(cls, policy_arg: str | Path, *, action_clip_override: float | None = None) -> "TrotDeploymentContract":
        policy_path = resolve_policy_path(policy_arg)
        if not policy_path.is_file():
            raise FileNotFoundError(f"exported policy not found: {policy_path}")
        run_dir = policy_path.parent.parent if policy_path.parent.name == "exported" else policy_path.parent
        env_yaml = run_dir / "params" / "env.yaml"
        agent_yaml = run_dir / "params" / "agent.yaml"
        for required_path in (env_yaml, agent_yaml):
            if not required_path.is_file():
                raise FileNotFoundError(f"refusing hardware deployment without exact run config: {required_path}")

        env = parse_isaaclab_yaml(env_yaml)
        agent = parse_isaaclab_yaml(agent_yaml)
        sim_dt = float(_required(env, "sim.dt", (int, float)))
        decimation = int(_required(env, "decimation", (int, float)))
        if sim_dt <= 0.0 or decimation <= 0:
            raise ValueError("sim.dt and decimation must be positive")

        # --- action / joint config (same hard rules as go2, all pass for trot1) ---
        action_joint_names = tuple(_required(env, "actions.joint_pos.joint_names", list))
        if action_joint_names != JOINT_ORDER:
            raise ValueError(f"action joint order mismatch: trained={action_joint_names}, deploy={JOINT_ORDER}")
        if _required(env, "actions.joint_pos.use_default_offset", bool) is not True:
            raise ValueError("trot1 deployment requires actions.joint_pos.use_default_offset=true")
        if _required(env, "actions.joint_pos.preserve_order", bool) is not True:
            raise ValueError("trot1 deployment requires actions.joint_pos.preserve_order=true")
        if float(_required(env, "actions.joint_pos.offset", (int, float))) != 0.0:
            raise ValueError("trot1 deployment requires actions.joint_pos.offset=0")
        if _required(env, "actions.joint_pos.class_type", str) != "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction":
            raise ValueError("trot1 deployment only supports JointPositionAction")

        # --- observation order (trot_* names, not crawl_*) ---
        observation_order = tuple(
            name
            for name in _required(env, "@order:observations.policy", list)
            if env.get(f"observations.policy.{name}.func") is not None
        )
        if observation_order != EXPECTED_OBS_ORDER:
            raise ValueError(
                f"trot1 observation order mismatch: trained={observation_order}, deploy={EXPECTED_OBS_ORDER}"
            )
        for term, function in (
            ("joint_pos", "isaaclab.envs.mdp.observations:joint_pos_rel"),
            ("joint_vel", "isaaclab.envs.mdp.observations:joint_vel_rel"),
        ):
            prefix = f"observations.policy.{term}"
            if _required(env, prefix + ".func", str) != function:
                raise ValueError(f"unsupported {term} observation function")
            observed_names = tuple(_required(env, prefix + ".params.asset_cfg.joint_names", list))
            if observed_names != JOINT_ORDER:
                raise ValueError(f"{term} observation joint order mismatch: {observed_names}")
            if _required(env, prefix + ".params.asset_cfg.preserve_order", bool) is not True:
                raise ValueError(f"{term} observation must preserve joint order")

        defaults = tuple(
            float(_required(env, f"scene.robot.init_state.joint_pos.{name}", (int, float))) for name in JOINT_ORDER
        )
        action_scale = float(_required(env, "actions.joint_pos.scale", (int, float)))
        if action_scale <= 0.0:
            raise ValueError("actions.joint_pos.scale must be positive")

        trained_clip = agent.get("clip_actions")
        if action_clip_override is not None:
            action_clip = float(action_clip_override)
            clip_source = "cli_override"
        elif trained_clip is None:
            action_clip = UNCLIPPED_RUNNER_DEPLOYMENT_CLIP
            clip_source = "explicit_deployment_guard_for_unclipped_runner"
        else:
            action_clip = float(trained_clip)
            clip_source = "agent.yaml"
        if not math.isfinite(action_clip) or action_clip <= 0.0:
            raise ValueError("deployment action clip must be finite and positive")

        gait = TrotGaitConfig(**{leaf: _gate_value(env, leaf) for leaf in GAIT_LEAFS})

        policy_sha256 = _sha256(policy_path)
        env_sha256 = _sha256(env_yaml)
        agent_sha256 = _sha256(agent_yaml)
        values = {
            "policy_sha256": policy_sha256,
            "env_sha256": env_sha256,
            "agent_sha256": agent_sha256,
            "joint_order": JOINT_ORDER,
            "joint_specs": tuple(asdict(spec) for spec in JOINT_SPECS),
            "train_default_rad": defaults,
            "sim_dt": sim_dt,
            "decimation": decimation,
            "action_scale_rad": action_scale,
            "action_clip": action_clip,
            "stiffness": _consistent_under(env, "scene.robot.actuators.", "stiffness"),
            "damping": _consistent_under(env, "scene.robot.actuators.", "damping"),
            "effort_limit": _consistent_under(env, "scene.robot.actuators.", "effort_limit"),
            "saturation_effort": _consistent_under(env, "scene.robot.actuators.", "saturation_effort"),
            "velocity_limit": _consistent_under(env, "scene.robot.actuators.", "velocity_limit"),
            "gait": asdict(gait),
        }
        fingerprint = hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
        ).hexdigest()[:16]
        return cls(
            policy_path=policy_path,
            run_dir=run_dir,
            env_yaml=env_yaml,
            agent_yaml=agent_yaml,
            policy_sha256=policy_sha256,
            env_sha256=env_sha256,
            agent_sha256=agent_sha256,
            joint_order=JOINT_ORDER,
            joint_specs=JOINT_SPECS,
            train_default_rad=defaults,
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            sim_dt=sim_dt,
            decimation=decimation,
            policy_rate_hz=1.0 / (sim_dt * decimation),
            action_scale_rad=action_scale,
            action_clip=action_clip,
            stiffness=values["stiffness"],
            damping=values["damping"],
            effort_limit=values["effort_limit"],
            saturation_effort=values["saturation_effort"],
            velocity_limit=values["velocity_limit"],
            gait=gait,
            action_clip_source=clip_source,
            fingerprint=fingerprint,
        )

    def summary_lines(self) -> Iterable[str]:
        g = self.gait
        yield f"[CONTRACT] id={self.fingerprint} policy={self.policy_path}"
        yield f"[CONTRACT] policy_sha256={self.policy_sha256}"
        yield f"[CONTRACT] env={self.env_yaml} sha256={self.env_sha256}"
        yield f"[CONTRACT] agent={self.agent_yaml} sha256={self.agent_sha256}"
        yield (
            f"[CONTRACT] rate={self.policy_rate_hz:.3f}Hz action_scale={self.action_scale_rad:.6f}rad "
            f"action_clip=+/-{self.action_clip:.3f}({self.action_clip_source}) kp={self.stiffness:.3f} kd={self.damping:.3f}"
        )
        yield (
            f"[CONTRACT] gait(speed-conditioned) deadband={g.command_deadband:.3f} "
            f"freq=[{g.min_frequency_hz:.2f},{g.max_frequency_hz:.2f}]Hz "
            f"speed=[{g.min_equivalent_speed:.2f},{g.max_equivalent_speed:.2f}]m/s "
            f"duty=[{g.high_speed_duty_factor:.2f},{g.low_speed_duty_factor:.2f}] "
            f"swing={g.swing_height:.3f}m yaw_r={g.yaw_equivalent_radius:.3f}"
        )
        yield (
            f"[CONTRACT] actuator effort={self.effort_limit:.3f}Nm saturation={self.saturation_effort:.3f}Nm "
            f"velocity={self.velocity_limit:.3f}rad/s"
        )
