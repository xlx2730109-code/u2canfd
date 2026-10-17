from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


JOINT_ORDER = (
    "FL_thigh",
    "FL_calf",
    "FR_thigh",
    "FR_calf",
    "RL_thigh",
    "RL_calf",
    "RR_thigh",
    "RR_calf",
)
OBSERVATION_DIM = 50
ACTION_DIM = 8
UNCLIPPED_RUNNER_DEPLOYMENT_CLIP = 3.5


@dataclass(frozen=True)
class JointSpec:
    name: str
    channel: int
    can_id: int
    master_id: int
    sim_to_motor: float


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


def resolve_policy_path(policy_arg: str | Path) -> Path:
    path = Path(policy_arg).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        candidates = sorted(
            path.glob("*/exported/policy.pt"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if path.name == "exported" and (path / "policy.pt").is_file():
            candidates.insert(0, path / "policy.pt")
        if candidates:
            return candidates[0].resolve()
    return path.resolve()


def _scalar(raw: str) -> Any:
    value = raw.strip()
    if not value or value.startswith("!!python/object") or value.startswith("!!python/tuple"):
        return _Container
    if value in ("null", "None", "~"):
        return None
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value) if value.lstrip("+-").isdigit() else float(value)
    except ValueError:
        return value.strip("'\"")


class _Container:
    pass


def parse_isaaclab_yaml(path: Path) -> dict[str, Any]:
    """Parse the indentation paths needed from Isaac Lab's Python-tagged YAML."""
    nodes: dict[str, Any] = {}
    stack: list[tuple[int, str]] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        content = raw_line.split(" #", 1)[0].rstrip()
        stripped = content.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(content) - len(content.lstrip(" "))
        if stripped.startswith("- "):
            # PyYAML emits block sequences at the same indentation as their key.
            while stack and indent < stack[-1][0]:
                stack.pop()
            if not stack:
                continue
            key = ".".join(item[1] for item in stack)
            nodes.setdefault(key, []).append(_scalar(stripped[2:]))
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        parent = ".".join(item[1] for item in stack)
        order_key = "@order:" + parent
        nodes.setdefault(order_key, []).append(key.strip())
        path_parts = [item[1] for item in stack] + [key.strip()]
        dotted = ".".join(path_parts)
        value = _scalar(raw_value)
        if value is _Container:
            stack.append((indent, key.strip()))
        else:
            nodes[dotted] = value
    return nodes


def _required(nodes: Mapping[str, Any], key: str, expected_type: type | tuple[type, ...] | None = None) -> Any:
    if key not in nodes:
        raise ValueError(f"training contract is missing '{key}'")
    value = nodes[key]
    if expected_type is not None and not isinstance(value, expected_type):
        raise ValueError(f"training contract '{key}' has invalid value {value!r}")
    return value


def _consistent_under(nodes: Mapping[str, Any], prefix: str, leaf: str) -> float:
    values = [float(value) for key, value in nodes.items() if key.startswith(prefix) and key.endswith("." + leaf)]
    if not values:
        raise ValueError(f"training contract has no '{leaf}' below '{prefix}'")
    first = values[0]
    if any(not math.isclose(value, first, rel_tol=0.0, abs_tol=1e-9) for value in values[1:]):
        raise ValueError(f"training contract has inconsistent '{leaf}' values below '{prefix}': {values}")
    return first


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DeploymentContract:
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
    gait_frequency_hz: float
    gait_duty_factor: float
    gait_swing_height: float
    command_deadband: float
    fingerprint: str

    @classmethod
    def load(cls, policy_arg: str | Path, *, action_clip_override: float | None = None) -> "DeploymentContract":
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

        action_joint_names = tuple(_required(env, "actions.joint_pos.joint_names", list))
        if action_joint_names != JOINT_ORDER:
            raise ValueError(f"action joint order mismatch: trained={action_joint_names}, deploy={JOINT_ORDER}")
        if _required(env, "actions.joint_pos.use_default_offset", bool) is not True:
            raise ValueError("go2-10 deployment requires actions.joint_pos.use_default_offset=true")
        if _required(env, "actions.joint_pos.preserve_order", bool) is not True:
            raise ValueError("go2-10 deployment requires actions.joint_pos.preserve_order=true")
        if float(_required(env, "actions.joint_pos.offset", (int, float))) != 0.0:
            raise ValueError("go2-10 deployment requires actions.joint_pos.offset=0")
        if _required(env, "actions.joint_pos.class_type", str) != "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction":
            raise ValueError("go2-10 deployment only supports JointPositionAction")

        expected_observation_order = (
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
            "crawl_phase",
            "crawl_leg_phase",
            "desired_contacts",
            "gait_params",
        )
        observation_order = tuple(
            name
            for name in _required(env, "@order:observations.policy", list)
            if env.get(f"observations.policy.{name}.func") is not None
        )
        if observation_order != expected_observation_order:
            raise ValueError(
                f"policy observation order mismatch: trained={observation_order}, deploy={expected_observation_order}"
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
            "gait_frequency_hz": _consistent_under(env, "observations.policy.", "frequency_hz"),
            "gait_duty_factor": _consistent_under(env, "observations.policy.", "duty_factor"),
            "gait_swing_height": _consistent_under(env, "observations.policy.", "swing_height"),
            "command_deadband": _consistent_under(env, "observations.policy.", "command_deadband"),
        }
        fingerprint_payload = json.dumps(values, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
        fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()[:16]
        return cls(
            policy_path=policy_path,
            run_dir=run_dir,
            env_yaml=env_yaml,
            agent_yaml=agent_yaml,
            policy_sha256=policy_sha256,
            env_sha256=env_sha256,
            agent_sha256=agent_sha256,
            joint_order=values["joint_order"],
            joint_specs=JOINT_SPECS,
            train_default_rad=values["train_default_rad"],
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            policy_rate_hz=1.0 / (sim_dt * decimation),
            sim_dt=values["sim_dt"],
            decimation=values["decimation"],
            action_scale_rad=values["action_scale_rad"],
            action_clip=values["action_clip"],
            stiffness=values["stiffness"],
            damping=values["damping"],
            effort_limit=values["effort_limit"],
            saturation_effort=values["saturation_effort"],
            velocity_limit=values["velocity_limit"],
            gait_frequency_hz=values["gait_frequency_hz"],
            gait_duty_factor=values["gait_duty_factor"],
            gait_swing_height=values["gait_swing_height"],
            command_deadband=values["command_deadband"],
            action_clip_source=clip_source,
            fingerprint=fingerprint,
        )

    def summary_lines(self) -> Iterable[str]:
        yield f"[CONTRACT] id={self.fingerprint} policy={self.policy_path}"
        yield f"[CONTRACT] policy_sha256={self.policy_sha256}"
        yield f"[CONTRACT] env={self.env_yaml} sha256={self.env_sha256}"
        yield f"[CONTRACT] agent={self.agent_yaml} sha256={self.agent_sha256}"
        yield (
            f"[CONTRACT] rate={self.policy_rate_hz:.3f}Hz action_scale={self.action_scale_rad:.6f}rad "
            f"action_clip=+/-{self.action_clip:.3f}({self.action_clip_source}) kp={self.stiffness:.3f} kd={self.damping:.3f}"
        )
        yield (
            f"[CONTRACT] gait frequency={self.gait_frequency_hz:.3f}Hz duty={self.gait_duty_factor:.3f} "
            f"swing_height={self.gait_swing_height:.3f}m deadband={self.command_deadband:.3f}"
        )
        yield (
            f"[CONTRACT] actuator effort={self.effort_limit:.3f}Nm saturation={self.saturation_effort:.3f}Nm "
            f"velocity={self.velocity_limit:.3f}rad/s"
        )
