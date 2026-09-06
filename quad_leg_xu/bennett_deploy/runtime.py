from __future__ import annotations

import argparse
import ctypes
import hashlib
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from .async_csv import AsyncCsvLogger
from .contract import DeploymentContract
from .gait import crawl_phase_terms
from .imu import DMImuSerialReader, FirstOrderLowPass, projected_gravity_from_rpy_deg
from .input import KeyboardCommandVelocity
from .policy import PolicyPipeline


DEFAULT_POLICY = r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_go2\quad_leg_go2-10\flat\2026-07-19_18-41-26\exported\policy.pt"

DM_SDK_PROFILE_ENV = "BENNETT_DM_SDK_PROFILE"
DM_SDK_PROFILES = {
    "original": ("dlls", "DB7FC43D0D60B847479DCB58CFBE13E524B0CA43C5DFA847D14E760BC0E2A8A7"),
    "low_latency": (
        "dlls_lowlatency",
        "252FAEFDC70F1C52D61F93E74E29B07F871F00230E75985D03CAE7C43B4AF15E",
    ),
}


def _verify_dm_sdk_profile(profile: str) -> tuple[Path | None, str | None]:
    """Resolve and verify the selected Windows SDK without opening USB/CAN."""
    if sys.platform != "win32":
        return None, None
    directory_name, expected_sha256 = DM_SDK_PROFILES[profile]
    root_dir = Path(__file__).resolve().parents[2]
    dll_dir = root_dir / directory_name
    dm_dll = dll_dir / "libdm_device.dll"
    required = (
        "libwinpthread-1.dll",
        "libgcc_s_seh-1.dll",
        "libstdc++-6.dll",
        "libusb-1.0.dll",
        "libdm_device.dll",
    )
    missing = [name for name in required if not (dll_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"DM SDK profile {profile!r} is incomplete in {dll_dir}: {missing}")
    actual_sha256 = hashlib.sha256(dm_dll.read_bytes()).hexdigest().upper()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"DM SDK profile {profile!r} hash mismatch: expected={expected_sha256} "
            f"actual={actual_sha256} path={dm_dll}"
        )
    return dll_dir, actual_sha256


class _WindowsTimerPeriod:
    """Process-scoped Windows timer resolution request, paired on shutdown."""

    def __init__(self, period_ms: int = 1):
        self.period_ms = int(period_ms)
        self._winmm = None
        self._active = False

    def begin(self) -> None:
        if sys.platform != "win32" or self._active:
            return
        self._winmm = ctypes.WinDLL("winmm")
        self._winmm.timeBeginPeriod.argtypes = [ctypes.c_uint]
        self._winmm.timeBeginPeriod.restype = ctypes.c_uint
        self._winmm.timeEndPeriod.argtypes = [ctypes.c_uint]
        self._winmm.timeEndPeriod.restype = ctypes.c_uint
        result = int(self._winmm.timeBeginPeriod(self.period_ms))
        if result != 0:
            self._winmm = None
            raise RuntimeError(f"timeBeginPeriod({self.period_ms}) failed: MMRESULT={result}")
        self._active = True
        print(f"[TIMER] timeBeginPeriod({self.period_ms}) rc=0", file=sys.stderr)

    def close(self) -> None:
        if not self._active or self._winmm is None:
            return
        result = int(self._winmm.timeEndPeriod(self.period_ms))
        self._active = False
        self._winmm = None
        print(f"[TIMER] timeEndPeriod({self.period_ms}) rc={result}", file=sys.stderr)


class StartupPhase(str, Enum):
    RAMP_DEFAULT = "ramp_default"
    WARMUP = "warmup"
    WAIT_ARM = "wait_arm"
    POLICY = "policy"


@dataclass(frozen=True)
class ImuTerms:
    gyro_raw: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    projected_gravity_raw: tuple[float, float, float]
    base_ang_vel_obs: tuple[float, float, float]
    projected_gravity_obs: tuple[float, float, float]
    gyro_age_s: float
    rpy_age_s: float


def _parse_values(text: str, count: int, name: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain {count} finite comma-separated values")
    return values


def _fmt(values: Sequence[float], digits: int = 3) -> str:
    return "[" + ", ".join(f"{float(value):+.{digits}f}" for value in values) + "]"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bennett go2-10 config-driven native hardware deployment")
    parser.add_argument("--policy", default=DEFAULT_POLICY, help="Exact policy.pt or experiment directory")
    parser.add_argument(
        "--policy-kind",
        choices=("go2", "trot1", "free"),
        default="go2",
        help="Deployment adapter: go2-10 contract (default), trot1 speed-conditioned gait, or free-gait (33-dim, no gait terms)",
    )
    parser.add_argument("--check_only", action="store_true", help="Validate contract and TorchScript without IMU or motors")
    parser.add_argument("--dry_run", action="store_true", help="Read IMU and run policy without opening motors")
    parser.add_argument(
        "--dm_sdk_profile",
        choices=tuple(DM_SDK_PROFILES),
        default="original",
        help="Windows DM SDK profile; Go2-11 selects the verified low_latency profile by default",
    )
    parser.add_argument("--imu_port", default="COM6")
    parser.add_argument("--imu_baudrate", type=int, default=921600)
    parser.add_argument(
        "--output_scale",
        type=float,
        default=1.0,
        help="Residual multiplier; 1.0 is the verified native policy behavior, lower values are staged-test overrides",
    )
    parser.add_argument("--action_clip", type=float, default=None, help="Explicit override; otherwise use agent.yaml/guard")
    parser.add_argument("--target_rate_limit_deg_s", type=float, default=60000.0)
    parser.add_argument(
        "--software_effort_limit_nm",
        type=float,
        default=0.0,
        help=(
            "Optional MIT-PD effort envelope implemented by limiting the sent position target from live "
            "position/velocity feedback; 0 disables it. Use the training actuator effort_limit for Sim2Real."
        ),
    )
    parser.add_argument("--kp", type=float, default=None, help="Explicit hardware override; default is exact env.yaml value")
    parser.add_argument("--kd", type=float, default=None, help="Explicit hardware override; default is exact env.yaml value")
    parser.add_argument("--command_x", type=float, default=None)
    parser.add_argument("--command_y", type=float, default=0.0)
    parser.add_argument("--command_yaw", type=float, default=0.0)
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument("--auto_arm", action="store_true")
    parser.add_argument("--keyboard_step_x", type=float, default=0.02)
    parser.add_argument(
        "--keyboard_min_nonzero_x",
        type=float,
        default=0.0,
        help="Optional trained minimum |vx|; zero disables near-zero ladder skipping",
    )
    parser.add_argument("--keyboard_step_y", type=float, default=0.02)
    parser.add_argument("--keyboard_step_yaw", type=float, default=0.10)
    parser.add_argument("--keyboard_min_x", type=float, default=-0.18)
    parser.add_argument("--keyboard_max_x", type=float, default=0.18)
    parser.add_argument("--keyboard_min_y", type=float, default=-0.10)
    parser.add_argument("--keyboard_max_y", type=float, default=0.10)
    parser.add_argument("--keyboard_min_yaw", type=float, default=-0.50)
    parser.add_argument("--keyboard_max_yaw", type=float, default=0.50)
    parser.add_argument("--active_joints", choices=("all", "rr"), default="all")
    parser.add_argument("--default_mode", choices=("motor_zero", "current_pose"), default="motor_zero")
    parser.add_argument("--motor_zero_positions", default="0,0,0,0,0,0,0,0")
    parser.add_argument("--default_rate_limit_deg_s", type=float, default=20.0)
    parser.add_argument("--default_reached_threshold_deg", type=float, default=2.0)
    parser.add_argument("--default_reached_hold_s", type=float, default=0.5)
    parser.add_argument("--default_timeout_s", type=float, default=6.0)
    parser.add_argument("--warmup_s", type=float, default=2.0)
    parser.add_argument("--duration_s", type=float, default=0.0)
    parser.add_argument("--tau_limit", type=float, default=18.0)
    parser.add_argument("--motor_feedback_timeout_s", type=float, default=0.50)
    parser.add_argument("--imu_timeout_s", type=float, default=0.10)
    parser.add_argument("--imu_cutoff_hz", type=float, default=12.0)
    parser.add_argument("--joint_vel_cutoff_hz", type=float, default=12.0)
    parser.add_argument("--gyro_signs", default="1,1,1")
    parser.add_argument("--gravity_signs", default="1,1,1")
    parser.add_argument("--gyro_unit", choices=("rad_s", "deg_s"), default="rad_s")
    parser.add_argument("--roll_offset_deg", type=float, default=0.0)
    parser.add_argument("--pitch_offset_deg", type=float, default=0.0)
    parser.add_argument("--yaw_offset_deg", type=float, default=0.0)
    parser.add_argument("--control_rate_hz", type=float, default=1000.0)
    parser.add_argument("--log_csv", default=None)
    parser.add_argument("--overwrite_log", action="store_true", help="Allow replacing an existing CSV log")
    parser.add_argument(
        "--log_csv_every",
        type=int,
        default=1,
        help="Log every N policy periods; unlike the legacy script this is not every 1 kHz control loop",
    )
    parser.add_argument("--diag_sequence", action="store_true")
    parser.add_argument("--diag_walk_only", action="store_true")
    parser.add_argument("--diag_stage_s", type=float, default=5.0)
    parser.add_argument("--diag_repeat_walk", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "output_scale",
        "target_rate_limit_deg_s",
        "default_rate_limit_deg_s",
        "default_reached_threshold_deg",
        "default_reached_hold_s",
        "default_timeout_s",
        "tau_limit",
        "motor_feedback_timeout_s",
        "imu_timeout_s",
        "control_rate_hz",
        "diag_stage_s",
    )
    for name in positive:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name} must be finite and > 0")
    if args.output_scale > 1.0:
        raise ValueError("--output_scale must be <= 1")
    if not math.isfinite(args.software_effort_limit_nm) or args.software_effort_limit_nm < 0.0:
        raise ValueError("--software_effort_limit_nm must be finite and >= 0")
    if args.warmup_s < 0.0 or args.duration_s < 0.0:
        raise ValueError("--warmup_s and --duration_s must be >= 0")
    if args.imu_cutoff_hz < 0.0 or args.joint_vel_cutoff_hz < 0.0:
        raise ValueError("filter cutoffs must be >= 0")
    if args.log_csv_every < 1:
        raise ValueError("--log_csv_every must be >= 1")


class ImuObservationSource:
    def __init__(self, args: argparse.Namespace, policy_dt: float):
        self.args = args
        self.gyro_signs = _parse_values(args.gyro_signs, 3, "--gyro_signs")
        self.gravity_signs = _parse_values(args.gravity_signs, 3, "--gravity_signs")
        self.gyro_filter = FirstOrderLowPass(args.imu_cutoff_hz, policy_dt)
        self.gravity_filter = FirstOrderLowPass(args.imu_cutoff_hz, policy_dt)
        self.reader = DMImuSerialReader(args.imu_port, args.imu_baudrate, exit_setting_mode=True)
        if not self.reader.wait_ready(timeout_s=2.0):
            diagnostics = self.reader.diagnostics()
            self.reader.close()
            raise RuntimeError(f"IMU did not provide gyro/euler frames on {args.imu_port}: {diagnostics}")

    def sample(self) -> ImuTerms:
        gyro_raw, gyro_age = self.reader.get_with_age(2)
        rpy_raw, rpy_age = self.reader.get_with_age(3)
        if gyro_raw is None or rpy_raw is None:
            raise RuntimeError("missing IMU gyro/euler data")
        if gyro_age > self.args.imu_timeout_s or rpy_age > self.args.imu_timeout_s:
            raise RuntimeError(
                f"stale IMU data: gyro_age={gyro_age:.3f}s rpy_age={rpy_age:.3f}s "
                f"limit={self.args.imu_timeout_s:.3f}s"
            )
        gyro = [float(gyro_raw[index]) * self.gyro_signs[index] for index in range(3)]
        if self.args.gyro_unit == "deg_s":
            gyro = [math.radians(value) for value in gyro]
        rpy = (
            float(rpy_raw[0]) - self.args.roll_offset_deg,
            float(rpy_raw[1]) - self.args.pitch_offset_deg,
            float(rpy_raw[2]) - self.args.yaw_offset_deg,
        )
        gravity = projected_gravity_from_rpy_deg(*rpy)
        gravity = tuple(gravity[index] * self.gravity_signs[index] for index in range(3))
        return ImuTerms(
            gyro_raw=tuple(gyro),
            rpy_deg=rpy,
            projected_gravity_raw=gravity,
            base_ang_vel_obs=tuple(self.gyro_filter.update(gyro)),
            projected_gravity_obs=tuple(self.gravity_filter.update(gravity)),
            gyro_age_s=gyro_age,
            rpy_age_s=rpy_age,
        )

    def diagnostics(self) -> dict:
        return self.reader.diagnostics()

    def close(self) -> None:
        self.reader.close()


class DeploymentRunner:
    def __init__(self, args: argparse.Namespace, contract: DeploymentContract, stop_event: threading.Event, policy_kind: str = "go2"):
        self.args = args
        self.contract = contract
        self.stop_event = stop_event
        self.policy_kind = policy_kind
        if policy_kind == "trot1":
            from .trot1_policy import TrotPolicyPipeline

            self.pipeline = TrotPolicyPipeline(
                contract,
                output_scale=args.output_scale,
                target_rate_limit_deg_s=args.target_rate_limit_deg_s,
            )
        elif policy_kind == "free":
            from .free_gait_policy import FreeGaitPolicyPipeline

            self.pipeline = FreeGaitPolicyPipeline(
                contract,
                output_scale=args.output_scale,
                target_rate_limit_deg_s=args.target_rate_limit_deg_s,
            )
        else:
            self.pipeline = PolicyPipeline(
                contract,
                output_scale=args.output_scale,
                target_rate_limit_deg_s=args.target_rate_limit_deg_s,
            )
        self.kp = contract.stiffness if args.kp is None else float(args.kp)
        self.kd = contract.damping if args.kd is None else float(args.kd)
        if not math.isfinite(self.kp) or not 0.0 <= self.kp <= 500.0:
            raise ValueError("MIT kp must be finite and within [0, 500]")
        if not math.isfinite(self.kd) or not 0.0 <= self.kd <= 5.0:
            raise ValueError("MIT kd must be finite and within [0, 5]")
        if args.control_rate_hz < contract.policy_rate_hz:
            raise ValueError("control rate must not be lower than the trained policy rate")
        self.active_names = (
            contract.joint_order if args.active_joints == "all" else ("RR_thigh", "RR_calf")
        )
        initial_x = 0.0 if args.keyboard and args.command_x is None else (0.10 if args.command_x is None else args.command_x)
        self.command = [float(initial_x), float(args.command_y), float(args.command_yaw)]
        self.keyboard = None
        if args.keyboard:
            self.keyboard = KeyboardCommandVelocity(
                stop_event=stop_event,
                initial_x=self.command[0],
                initial_y=self.command[1],
                initial_yaw=self.command[2],
                min_x=args.keyboard_min_x,
                max_x=args.keyboard_max_x,
                min_y=args.keyboard_min_y,
                max_y=args.keyboard_max_y,
                min_yaw=args.keyboard_min_yaw,
                max_yaw=args.keyboard_max_yaw,
                step_x=args.keyboard_step_x,
                step_y=args.keyboard_step_y,
                step_yaw=args.keyboard_step_yaw,
                min_nonzero_x=args.keyboard_min_nonzero_x,
                auto_arm=args.auto_arm,
            )
        self.logger = AsyncCsvLogger(args.log_csv, overwrite=args.overwrite_log) if args.log_csv else None
        if self.logger is not None:
            print(f"[LOG-CSV] async policy-rate diagnostics -> {self.logger.path}")
        self.imu: ImuObservationSource | None = None
        self.bus = None
        self.dm_sdk_dir, self.dm_sdk_sha256 = _verify_dm_sdk_profile(args.dm_sdk_profile)
        self.windows_timer_period = _WindowsTimerPeriod(1)
        self.control_overruns = 0
        self.max_control_late_s = 0.0

    def print_contract(self) -> None:
        for line in self.contract.summary_lines():
            print(line)
        if self.policy_kind == "trot1":
            obs_tail = "trot_phase(2), trot_leg_phase(8), desired_contacts(4), gait_params(3)"
        elif self.policy_kind == "free":
            obs_tail = "(no gait terms; gait is emergent)"
        else:
            obs_tail = "crawl_phase/crawl_leg_phase(2/8), desired_contacts(4), gait_params(3)"
        print("[OBS] base_ang_vel(3), projected_gravity(3), velocity_command(3), joint_pos_rel(8), "
              f"joint_vel_rel(8), last_action(8), {obs_tail}")
        print("[JOINT-ORDER] " + ", ".join(self.contract.joint_order))
        print(
            "[TRAIN-DEFAULT] "
            + ", ".join(
                f"{name}:{math.degrees(self.contract.train_default_rad[index]):+.1f}deg"
                for index, name in enumerate(self.contract.joint_order)
            )
        )
        print(
            f"[RUNTIME] kp={self.kp:.3f} kd={self.kd:.3f} output_scale={self.args.output_scale:.3f} "
            f"active={self.args.active_joints} control_rate={self.args.control_rate_hz:.1f}Hz "
            f"effort_envelope={self.args.software_effort_limit_nm:.3f}Nm "
            f"tau_stop={self.args.tau_limit:.3f}Nm"
        )
        if self.dm_sdk_dir is not None:
            print(
                f"[DM-SDK] profile={self.args.dm_sdk_profile} path={self.dm_sdk_dir} "
                f"libdm_sha256={self.dm_sdk_sha256}"
            )
        print(
            "[ACTION] q_cmd = motor_zero + sim_to_motor * "
            "(train_default + clipped_policy_action * action_scale * output_scale), "
            "then optional live-feedback MIT-PD effort limiting"
        )

    def preflight(self) -> dict[str, float]:
        result = self.pipeline.preflight()
        print(
            f"[PREFLIGHT] samples={int(result['samples'])} raw=[{result['raw_min']:+.3f}, {result['raw_max']:+.3f}] "
            f"abs_max={result['raw_abs_max']:.3f} clip_fraction={result['clip_fraction']:.4%} "
            f"max_residual={result['max_residual_deg']:.2f}deg"
        )
        if result["clip_fraction"] > 0.05:
            print("[PREFLIGHT] warning: synthetic gait clips more than 5% of action values", file=sys.stderr)
        return result

    def _poll_command(self) -> None:
        if self.keyboard is not None:
            self.command[:] = self.keyboard.poll()

    def _diag_command(self, policy_elapsed_s: float) -> str:
        if not self.args.diag_sequence:
            self._poll_command()
            return "manual"
        stages = (
            (("ground", 0.0), ("walk_0p04", 0.04), ("walk_0p08", 0.08))
            if self.args.diag_walk_only
            else (("air", 0.0), ("half_load", 0.0), ("ground", 0.0), ("walk_0p04", 0.04), ("walk_0p08", 0.08))
        )
        index = int(policy_elapsed_s // self.args.diag_stage_s)
        walk_start = len(stages) - 2
        if self.args.diag_repeat_walk and index >= walk_start:
            index = walk_start + ((index - walk_start) % 2)
        if index >= len(stages):
            self.command[:] = (0.0, 0.0, 0.0)
            return "done"
        name, command_x = stages[index]
        self.command[:] = (command_x, 0.0, 0.0)
        return name

    def _dry_run(self) -> None:
        assert self.imu is not None
        start = time.monotonic()
        next_policy = start
        count = 0
        while not self.stop_event.is_set():
            now = time.monotonic()
            if self.args.duration_s > 0.0 and now - start >= self.args.duration_s:
                break
            self._poll_command()
            if now >= next_policy:
                terms = self.imu.sample()
                armed = self.keyboard is None or self.keyboard.armed
                if armed:
                    step = self.pipeline.infer(
                        base_ang_vel=terms.base_ang_vel_obs,
                        projected_gravity=terms.projected_gravity_obs,
                        velocity_command=self.command,
                        joint_pos_rel=(0.0,) * 8,
                        joint_vel_rel=(0.0,) * 8,
                        phase_elapsed_s=now - start,
                    )
                    if count % max(1, int(self.contract.policy_rate_hz // 2)) == 0:
                        print(
                            f"[DRY-RUN] cmd={_fmt(self.command)} rpy={_fmt(terms.rpy_deg)} "
                            f"raw={_fmt(step.raw_action.tolist())} action={_fmt(step.action.tolist())}"
                        )
                next_policy += self.pipeline.policy_dt
                if next_policy < now - self.pipeline.policy_dt:
                    next_policy = now + self.pipeline.policy_dt
                count += 1
            time.sleep(0.001)

    def _feedback_to_policy(
        self,
        snapshot: Mapping[str, object],
        motor_zero: Mapping[str, float],
    ) -> tuple[list[float], list[float], list[float]]:
        sim_pos, rel_pos, sim_vel = [], [], []
        for index, spec in enumerate(self.contract.joint_specs):
            feedback = snapshot[spec.name]
            position = (feedback.position - motor_zero[spec.name]) / spec.sim_to_motor
            sim_pos.append(position)
            rel_pos.append(position - self.contract.train_default_rad[index])
            sim_vel.append(feedback.velocity / spec.sim_to_motor)
        return sim_pos, rel_pos, sim_vel

    def _motor_targets(self, target_sim) -> dict[str, float]:
        return {
            spec.name: self.motor_zero[spec.name] + spec.sim_to_motor * float(target_sim[index])
            for index, spec in enumerate(self.contract.joint_specs)
        }

    def _software_effort_limited_targets(
        self,
        targets: Mapping[str, float],
        snapshot: Mapping[str, object],
    ) -> dict[str, float]:
        """Match Isaac Lab's low-speed DCMotor effort cap using the MIT PD target.

        The hardware command is ``tau = kp * (q_des - q) - kd * dq`` because
        desired velocity and feed-forward torque are both zero.  Clamping that
        torque and solving back for ``q_des`` preserves the trained PD gains
        while reproducing the simulation's continuous effort envelope.
        """
        limit = float(self.args.software_effort_limit_nm)
        if limit <= 0.0:
            return dict(targets)
        if self.kp <= 0.0:
            raise RuntimeError("--software_effort_limit_nm requires kp > 0")
        limited = dict(targets)
        for name in self.active_names:
            state = snapshot[name]
            desired_tau = self.kp * (float(targets[name]) - state.position) - self.kd * state.velocity
            limited_tau = max(-limit, min(limit, desired_tau))
            limited[name] = state.position + (limited_tau + self.kd * state.velocity) / self.kp
        return limited

    def _validate_feedback(self, snapshot: Mapping[str, object], loop_count: int) -> None:
        for name in self.active_names:
            feedback = snapshot[name]
            if feedback.error not in (0, 1):
                raise RuntimeError(f"{name} motor error: {feedback.error}")
            if loop_count > 100 and feedback.feedback_age_s > self.args.motor_feedback_timeout_s:
                raise RuntimeError(
                    f"{name} feedback stale: age={feedback.feedback_age_s:.3f}s "
                    f"limit={self.args.motor_feedback_timeout_s:.3f}s"
                )
            if loop_count > 100 and abs(feedback.torque) > self.args.tau_limit:
                raise RuntimeError(f"{name} tau too high: {feedback.torque:.3f}Nm")

    def _log_rows(
        self,
        *,
        now: float,
        elapsed_s: float,
        loop_count: int,
        policy_step_count: int,
        startup_phase: StartupPhase,
        diag_stage: str,
        gait,
        raw_action,
        action,
        desired_offset,
        applied_offset,
        target_sim,
        rel_pos: Sequence[float],
        joint_vel_obs: Sequence[float],
        q_policy: Mapping[str, float],
        q_cmd: Mapping[str, float],
        feedback: Mapping[str, object],
        imu_terms: ImuTerms,
    ) -> None:
        if self.logger is None:
            return
        rows = []
        for index, name in enumerate(self.contract.joint_order):
            state = feedback[name]
            rows.append((
                now, elapsed_s, loop_count, policy_step_count, startup_phase.value, diag_stage,
                self.contract.fingerprint, *self.command, gait.phase, name, int(name in self.active_names),
                self.kp, self.kd, self.args.output_scale, self.contract.action_clip, self.contract.policy_rate_hz,
                self.args.target_rate_limit_deg_s, self.args.software_effort_limit_nm,
                float(raw_action[index]), float(action[index]),
                float(desired_offset[index]), float(applied_offset[index]), float(target_sim[index]),
                math.degrees(float(target_sim[index])), self.contract.train_default_rad[index], rel_pos[index],
                math.degrees(rel_pos[index]), q_policy[name], q_cmd[name], state.position, state.velocity,
                joint_vel_obs[index],
                state.torque, state.error, state.feedback_age_s, *imu_terms.base_ang_vel_obs,
                *imu_terms.projected_gravity_obs, imu_terms.gyro_age_s, imu_terms.rpy_age_s,
                gait.desired_contacts[index // 2],
            ))
        self.logger.submit(rows)

    def _hardware_run(self) -> None:
        os.environ[DM_SDK_PROFILE_ENV] = self.args.dm_sdk_profile
        if self.args.dm_sdk_profile == "low_latency":
            self.windows_timer_period.begin()
        from .dm_can import Go2MotorBus

        assert self.imu is not None
        self.bus = Go2MotorBus(self.contract.joint_specs)
        self.bus.wait_fresh_feedback(timeout_s=1.0)
        first_snapshot = self.bus.snapshot()
        configured_zero = _parse_values(self.args.motor_zero_positions, 8, "--motor_zero_positions")
        if self.args.default_mode == "motor_zero":
            self.motor_zero = dict(zip(self.contract.joint_order, configured_zero))
        else:
            self.motor_zero = {
                spec.name: first_snapshot[spec.name].position
                - spec.sim_to_motor * self.contract.train_default_rad[index]
                for index, spec in enumerate(self.contract.joint_specs)
            }
        startup_sim, _, _ = self._feedback_to_policy(first_snapshot, self.motor_zero)
        target_sim = self.pipeline.torch.tensor(startup_sim, dtype=self.pipeline.torch.float32)
        train_default = self.pipeline.train_default
        active_indices = self.pipeline.torch.tensor(
            [self.contract.joint_order.index(name) for name in self.active_names], dtype=self.pipeline.torch.long
        )
        q_policy = self._motor_targets(target_sim)
        q_cmd = self._software_effort_limited_targets(q_policy, first_snapshot)
        self._validate_feedback(first_snapshot, 0)
        # Hard invariant: the first non-zero-gain command is exactly the measured startup pose.
        self.bus.command(q_cmd, self.active_names, kp=self.kp, kd=self.kd)
        print(
            "[SAFE-START] startup_motor_pos_rad="
            + ", ".join(f"{name}:{first_snapshot[name].position:+.4f}" for name in self.contract.joint_order),
            file=sys.stderr,
        )
        print(
            "[SAFE-START] motor_zero_rad="
            + ", ".join(f"{name}:{self.motor_zero[name]:+.4f}" for name in self.contract.joint_order),
            file=sys.stderr,
        )
        print(
            f"[SAFE-START] first command=current pose; ramp={self.args.default_rate_limit_deg_s:.2f}deg/s "
            f"threshold={self.args.default_reached_threshold_deg:.2f}deg hold={self.args.default_reached_hold_s:.2f}s "
            f"warmup={self.args.warmup_s:.2f}s timeout={self.args.default_timeout_s:.2f}s",
            file=sys.stderr,
        )

        torch = self.pipeline.torch
        zeros = torch.zeros(8, dtype=torch.float32)
        raw_action = action = desired_offset = applied_offset = zeros.clone()
        phase = StartupPhase.RAMP_DEFAULT
        default_reached_since = None
        warmup_start = None
        policy_start = None
        next_policy = time.monotonic()
        next_log = next_policy
        policy_step_count = 0
        log_period = self.pipeline.policy_dt * self.args.log_csv_every
        joint_velocity_filter = FirstOrderLowPass(self.args.joint_vel_cutoff_hz, self.pipeline.policy_dt)
        joint_vel_obs = [0.0] * 8
        imu_terms = self.imu.sample()
        if self.policy_kind == "trot1":
            gait = self.pipeline.gait_clock.step(self.command)
        elif self.policy_kind == "free":
            gait = self.pipeline.placeholder_gait()
        else:
            gait = crawl_phase_terms(0.0, self.contract.gait_frequency_hz, self.contract.gait_duty_factor,
                                     self.contract.gait_swing_height, self.command, self.contract.command_deadband)
        diag_stage = "manual"
        start = time.monotonic()
        last_control = start
        period = 1.0 / self.args.control_rate_hz
        deadline = time.perf_counter()
        loop_count = 0

        while not self.stop_event.is_set():
            now = time.monotonic()
            elapsed = now - start
            if self.args.duration_s > 0.0 and elapsed >= self.args.duration_s:
                break
            loop_count += 1
            snapshot = self.bus.snapshot(now)
            _, rel_pos, sim_vel = self._feedback_to_policy(snapshot, self.motor_zero)
            control_dt = max(0.0, min(now - last_control, 0.02))
            last_control = now

            if phase is StartupPhase.RAMP_DEFAULT:
                max_delta = math.radians(self.args.default_rate_limit_deg_s) * control_dt
                target_sim += torch.clamp(train_default - target_sim, -max_delta, max_delta)
                self.pipeline.reset()
                raw_action = action = desired_offset = applied_offset = zeros.clone()
                target_error = torch.max(torch.abs((target_sim - train_default).index_select(0, active_indices))).item()
                real_error = max(abs(rel_pos[index]) for index in active_indices.tolist())
                if target_error <= math.radians(self.args.default_reached_threshold_deg) and real_error <= math.radians(self.args.default_reached_threshold_deg):
                    if default_reached_since is None:
                        default_reached_since = now
                    elif now - default_reached_since >= self.args.default_reached_hold_s:
                        phase = StartupPhase.WARMUP
                        warmup_start = now
                        target_sim = train_default.clone()
                        print(f"[SAFE-START] trained default reached; warmup {self.args.warmup_s:.2f}s", file=sys.stderr)
                else:
                    default_reached_since = None
                if elapsed > self.args.default_timeout_s:
                    raise RuntimeError(
                        f"default pose not reached within {self.args.default_timeout_s:.2f}s: "
                        f"real_err={math.degrees(real_error):.2f}deg target_err={math.degrees(target_error):.2f}deg"
                    )
            elif phase is StartupPhase.WARMUP:
                target_sim = train_default.clone()
                self.pipeline.reset()
                raw_action = action = desired_offset = applied_offset = zeros.clone()
                if warmup_start is not None and now - warmup_start >= self.args.warmup_s:
                    if self.keyboard is not None and not self.keyboard.armed:
                        phase = StartupPhase.WAIT_ARM
                        print("[SAFE-START] default ready; press Enter/Space to ARM", file=sys.stderr)
                    else:
                        phase = StartupPhase.POLICY
                        policy_start = now
                        next_policy = now
                        self.pipeline.reset()
                        print("[SAFE-START] policy enabled; gait phase reset to 0", file=sys.stderr)
            elif phase is StartupPhase.WAIT_ARM:
                target_sim = train_default.clone()
                self.pipeline.reset()
                raw_action = action = desired_offset = applied_offset = zeros.clone()
                self._poll_command()
                if self.keyboard is None or self.keyboard.armed:
                    phase = StartupPhase.POLICY
                    policy_start = now
                    next_policy = now
                    self.pipeline.reset()
                    print("[SAFE-START] policy ARMED; gait phase reset to 0", file=sys.stderr)
            elif now >= next_policy:
                assert policy_start is not None
                policy_elapsed = now - policy_start
                diag_stage = self._diag_command(policy_elapsed)
                imu_terms = self.imu.sample()
                joint_vel_obs = joint_velocity_filter.update(sim_vel)
                step = self.pipeline.infer(
                    base_ang_vel=imu_terms.base_ang_vel_obs,
                    projected_gravity=imu_terms.projected_gravity_obs,
                    velocity_command=self.command,
                    joint_pos_rel=rel_pos,
                    joint_vel_rel=joint_vel_obs,
                    phase_elapsed_s=policy_elapsed,
                )
                raw_action, action = step.raw_action, step.action
                desired_offset, applied_offset, target_sim, gait = (
                    step.desired_offset_rad,
                    step.applied_offset_rad,
                    step.target_sim_rad,
                    step.gait,
                )
                policy_step_count += 1
                next_policy += self.pipeline.policy_dt
                if next_policy < now - self.pipeline.policy_dt:
                    next_policy = now + self.pipeline.policy_dt

            q_policy = self._motor_targets(target_sim)
            q_cmd = self._software_effort_limited_targets(q_policy, snapshot)
            self._validate_feedback(snapshot, loop_count)
            self.bus.command(q_cmd, self.active_names, kp=self.kp, kd=self.kd)

            if self.logger is not None and now >= next_log:
                self._log_rows(
                    now=now, elapsed_s=elapsed, loop_count=loop_count, policy_step_count=policy_step_count,
                    startup_phase=phase, diag_stage=diag_stage, gait=gait, raw_action=raw_action, action=action,
                    desired_offset=desired_offset, applied_offset=applied_offset, target_sim=target_sim,
                    rel_pos=rel_pos, joint_vel_obs=joint_vel_obs, q_policy=q_policy, q_cmd=q_cmd,
                    feedback=snapshot, imu_terms=imu_terms,
                )
                next_log += log_period
                if next_log < now - log_period:
                    next_log = now + log_period

            if loop_count % 200 == 0:
                tau_max = max(abs(snapshot[name].torque) for name in self.active_names)
                print(
                    f"phase={phase.value} stage={diag_stage} cmd={_fmt(self.command)} gait={gait.phase:.3f} "
                    f"raw={_fmt(raw_action.tolist(), 2)} action={_fmt(action.tolist(), 2)} "
                    f"target_deg={_fmt([math.degrees(float(value)) for value in target_sim], 1)} "
                    f"tau_max={tau_max:.3f} imu_age_ms={max(imu_terms.gyro_age_s, imu_terms.rpy_age_s)*1000.0:.1f}"
                )

            deadline += period
            late = time.perf_counter() - deadline
            if late < 0.0:
                time.sleep(-late)
            else:
                self.control_overruns += 1
                self.max_control_late_s = max(self.max_control_late_s, late)
                if late > 5.0 * period:
                    deadline = time.perf_counter()

    def run(self) -> None:
        self.print_contract()
        self.preflight()
        if self.args.check_only:
            print("[CHECK-ONLY] contract and TorchScript validation passed; IMU and motors were not opened")
            return
        self.imu = ImuObservationSource(self.args, self.pipeline.policy_dt)
        print("[IMU] diagnostics=" + str(self.imu.diagnostics()))
        if self.args.dry_run:
            print("[DRY-RUN] motors will not be opened")
            self._dry_run()
        else:
            self._hardware_run()

    def close(self) -> None:
        errors = []
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception as exc:
                errors.append(f"motor close: {exc}")
        if self.imu is not None:
            try:
                print("[IMU] final diagnostics=" + str(self.imu.diagnostics()), file=sys.stderr)
                self.imu.close()
            except Exception as exc:
                errors.append(f"IMU close: {exc}")
        if self.logger is not None:
            try:
                print("[LOG-CSV] " + str(self.logger.close()), file=sys.stderr)
            except Exception as exc:
                errors.append(f"CSV close: {exc}")
        try:
            self.windows_timer_period.close()
        except Exception as exc:
            errors.append(f"timer close: {exc}")
        print(
            f"[TIMING] control_overruns={self.control_overruns} max_late_ms={self.max_control_late_s*1000.0:.3f}",
            file=sys.stderr,
        )
        if errors:
            print("[CLOSE] " + "; ".join(errors), file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = None
    stop_event = threading.Event()

    def request_stop(signum, frame):
        del frame
        print(f"\n[SIGNAL] {signum}; stopping", file=sys.stderr)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
    try:
        _validate_args(args)
        if args.policy_kind == "trot1":
            from .trot1_contract import TrotDeploymentContract

            contract = TrotDeploymentContract.load(args.policy, action_clip_override=args.action_clip)
        elif args.policy_kind == "free":
            from .free_gait_contract import FreeGaitDeploymentContract

            contract = FreeGaitDeploymentContract.load(args.policy, action_clip_override=args.action_clip)
        else:
            contract = DeploymentContract.load(args.policy, action_clip_override=args.action_clip)
        runner = DeploymentRunner(args, contract, stop_event, policy_kind=args.policy_kind)
        runner.run()
        return 0
    except KeyboardInterrupt:
        stop_event.set()
        return 130
    except Exception as exc:
        print(f"Error: deployment failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if runner is not None:
            runner.close()
