"""Free-gait policy pipeline: observation builder + MIT-position decode.

This is the free-gait sibling of `policy.py` / `trot1_policy.py`, sharing the
exact `infer` / `reset` / `preflight` interface so `runtime.py` can select it
with the `--policy-kind free` flag without touching the go2/trot1 paths.

There is no gait clock and no phase/desired_contact/gait_params observation:
the observation is exactly the base + command + joint + action block (33 dims).
The gait is emergent and lives only in the trained network weights.

`placeholder_gait()` returns a tiny object with the `phase` / `desired_contacts`
attributes that runtime's CSV logger and status line expect. For a free-gait
policy these carry no information (there is no commanded gait), so phase is 0
and desired contacts are all zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .free_gait_contract import FreeGaitDeploymentContract


@dataclass(frozen=True)
class FreeGaitTerms:
    phase: float = 0.0
    desired_contacts: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class FreeGaitPolicyStep:
    observation: object
    raw_action: object
    action: object
    desired_offset_rad: object
    applied_offset_rad: object
    target_sim_rad: object
    gait: FreeGaitTerms


class FreeGaitObservationBuilder:
    def __init__(self, contract: FreeGaitDeploymentContract, torch_module):
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
    ):
        values = (
            list(base_ang_vel)
            + list(projected_gravity)
            + list(velocity_command)
            + list(joint_pos_rel)
            + list(joint_vel_rel)
            + list(last_action.tolist())
        )
        if len(values) != self.contract.observation_dim:
            raise RuntimeError(f"observation length mismatch: {len(values)} != {self.contract.observation_dim}")
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("observation contains non-finite values")
        return self.torch.tensor(values, dtype=self.torch.float32).unsqueeze(0)


class FreeGaitPolicyPipeline:
    def __init__(
        self,
        contract: FreeGaitDeploymentContract,
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
        self.observations = FreeGaitObservationBuilder(contract, torch)
        self.last_action = torch.zeros(contract.action_dim, dtype=torch.float32)
        self.applied_offset = torch.zeros(contract.action_dim, dtype=torch.float32)
        self.train_default = torch.tensor(contract.train_default_rad, dtype=torch.float32)

    def placeholder_gait(self) -> FreeGaitTerms:
        return FreeGaitTerms()

    def reset(self) -> None:
        self.last_action.zero_()
        self.applied_offset.zero_()

    def infer(
        self,
        *,
        base_ang_vel: Sequence[float],
        projected_gravity: Sequence[float],
        velocity_command: Sequence[float],
        joint_pos_rel: Sequence[float],
        joint_vel_rel: Sequence[float],
        phase_elapsed_s: float,
    ) -> FreeGaitPolicyStep:
        del phase_elapsed_s  # no gait clock for a free-gait policy
        obs = self.observations.build(
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            velocity_command=velocity_command,
            joint_pos_rel=joint_pos_rel,
            joint_vel_rel=joint_vel_rel,
            last_action=self.last_action,
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
        return FreeGaitPolicyStep(obs, raw, action, desired, self.applied_offset.clone(), target, FreeGaitTerms())

    def preflight(self, *, command_x: float = 0.08) -> dict[str, float]:
        """Validate policy I/O over one second of zero-state inference (no hardware)."""
        saved_last = self.last_action.clone()
        saved_offset = self.applied_offset.clone()
        self.reset()
        raws = []
        clipped_values = 0
        samples = max(2, int(round(self.contract.policy_rate_hz)))
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
