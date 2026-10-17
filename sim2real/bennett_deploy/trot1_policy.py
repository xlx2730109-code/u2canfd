"""Trot1 policy pipeline: observation builder + MIT-position decode.

This is the trot1 sibling of `policy.py`, sharing the exact `infer` /
`reset` / `preflight` interface so `runtime.py` can switch to it with the
`--policy-kind trot1` flag without touching the go2 path.

The only behavioural differences from the go2 pipeline are:
  * gait is produced by the stateful, speed-conditioned `TrotGaitClock`
    (from trot1_gait.py) instead of the fixed-frequency `crawl_phase_terms`;
  * the observation vector is laid out for the trot1 policy: grouped leg
    phase (sin*4 then cos*4), trot_phase / trot_leg_phase / desired_contacts /
    gait_params in that position, 50 dims total.

NOTE (no per-joint action clip is mirrored, and that is deliberate): Isaac Lab's
`JointPositionAction.process_actions` clips the ABSOLUTE target
(`raw*scale + default_joint_pos`) to `actions.joint_pos.clip`. Given the trot1
defaults (thigh +-0.08, calf -0.16), action_scale=0.2 and action_clip=3.5
(offset = +-0.7 rad), the absolute target never leaves [thigh +-0.8]
[calf -0.9, +0.55] -- the sim clip never binds. `--output_scale` is hard-limited
to <=1, so the offset can't exceed 0.7 rad either. Mirroring the per-joint clip
would be a no-op and risk mis-clamping the offset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .trot1_contract import TrotDeploymentContract
from .trot1_gait import TrotGaitClock, TrotGaitTerms


@dataclass(frozen=True)
class TrotPolicyStep:
    observation: object
    raw_action: object
    action: object
    desired_offset_rad: object
    applied_offset_rad: object
    target_sim_rad: object
    gait: TrotGaitTerms


class TrotObservationBuilder:
    def __init__(self, contract: TrotDeploymentContract, torch_module):
        self.contract = contract
        self.torch = torch_module

    def build(
        self,
        *,
        base_ang_vel: Sequence[float],
        projected_gravity: Sequence[float],
        velocity_command: Sequence[float],
        joint_pos_rel: Sequence[float],
        joint_vel_rel: Sequence[float],
        last_action,
        gait: TrotGaitTerms,
    ):
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
            raise RuntimeError(f"observation length mismatch: {len(values)} != {self.contract.observation_dim}")
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("observation contains non-finite values")
        return self.torch.tensor(values, dtype=self.torch.float32).unsqueeze(0)


class TrotPolicyPipeline:
    def __init__(
        self,
        contract: TrotDeploymentContract,
        *,
        output_scale: float,
        target_rate_limit_deg_s: float,
    ):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("the deployment Python environment must provide torch") from exc
        self.torch = torch
        self.contract = contract
        self.output_scale = float(output_scale)
        self.target_rate_limit_rad_s = math.radians(float(target_rate_limit_deg_s))
        self.policy_dt = 1.0 / contract.policy_rate_hz
        self.policy = torch.jit.load(str(contract.policy_path), map_location="cpu")
        self.policy.eval()
        self.observations = TrotObservationBuilder(contract, torch)
        self.gait_clock = TrotGaitClock(contract.gait, self.policy_dt)
        self.last_action = torch.zeros(contract.action_dim, dtype=torch.float32)
        self.applied_offset = torch.zeros(contract.action_dim, dtype=torch.float32)
        self.train_default = torch.tensor(contract.train_default_rad, dtype=torch.float32)

    def reset(self) -> None:
        self.last_action.zero_()
        self.applied_offset.zero_()
        self.gait_clock.reset()

    def infer(
        self,
        *,
        base_ang_vel: Sequence[float],
        projected_gravity: Sequence[float],
        velocity_command: Sequence[float],
        joint_pos_rel: Sequence[float],
        joint_vel_rel: Sequence[float],
        phase_elapsed_s: float,
    ) -> TrotPolicyStep:
        del phase_elapsed_s  # the clock integrates itself via policy_dt
        gait = self.gait_clock.step(velocity_command)
        obs = self.observations.build(
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            velocity_command=velocity_command,
            joint_pos_rel=joint_pos_rel,
            joint_vel_rel=joint_vel_rel,
            last_action=self.last_action,
            gait=gait,
        )
        with self.torch.no_grad():
            raw = self.policy(obs).squeeze(0).to(self.torch.float32)
        if raw.numel() != self.contract.action_dim:
            raise RuntimeError(f"policy action length mismatch: {raw.numel()} != {self.contract.action_dim}")
        if not bool(self.torch.isfinite(raw).all()):
            raise RuntimeError(f"policy returned non-finite action: {raw.tolist()}")
        action = self.torch.clamp(raw, -self.contract.action_clip, self.contract.action_clip)
        desired = action * self.contract.action_scale_rad * self.output_scale
        max_delta = self.target_rate_limit_rad_s * self.policy_dt
        self.applied_offset += self.torch.clamp(desired - self.applied_offset, -max_delta, max_delta)
        target = self.train_default + self.applied_offset
        self.last_action = action.clone()
        return TrotPolicyStep(obs, raw, action, desired, self.applied_offset.clone(), target, gait)

    def preflight(self, *, command_x: float = 0.08) -> dict[str, float]:
        """Validate policy I/O over one gait cycle (clock-driven, no hardware)."""
        saved_last = self.last_action.clone()
        saved_offset = self.applied_offset.clone()
        self.reset()
        raws = []
        clipped_values = 0
        nominal_freq = (self.contract.gait.min_frequency_hz + self.contract.gait.max_frequency_hz) / 2.0
        samples = max(2, int(round(self.contract.policy_rate_hz / nominal_freq)))
        command = (float(command_x), 0.0, 0.0)
        try:
            for _ in range(samples):
                step = self.infer(
                    base_ang_vel=(0.0, 0.0, 0.0),
                    projected_gravity=(0.0, 0.0, -1.0),
                    velocity_command=command,
                    joint_pos_rel=(0.0,) * self.contract.action_dim,
                    joint_vel_rel=(0.0,) * self.contract.action_dim,
                    phase_elapsed_s=0.0,
                )
                raws.append(step.raw_action)
                clipped_values += int((self.torch.abs(step.raw_action) > self.contract.action_clip).sum().item())
            stacked = self.torch.stack(raws)
            return {
                "samples": float(samples),
                "raw_min": float(stacked.min().item()),
                "raw_max": float(stacked.max().item()),
                "raw_abs_max": float(stacked.abs().max().item()),
                "clip_fraction": clipped_values / float(stacked.numel()),
                "max_residual_deg": math.degrees(
                    self.contract.action_scale_rad * self.output_scale
                    * min(float(stacked.abs().max()), self.contract.action_clip)
                ),
            }
        finally:
            self.last_action = saved_last
            self.applied_offset = saved_offset
