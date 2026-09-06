"""Speed-conditioned trot clock for the deployment side.

Faithfully replicates the trot1 sim clock (`_schedule_from_env` in
quad_leg_trot1/mdp/gait.py) in scalar Python (no torch). This is the SWAPPABLE
gait module: if the gait is redesigned (e.g. dropped for an unconstrained RL
gait), replace this module and remove the four gait observation terms from the
obs builder -- the rest of the deployment is untouched.

Key layout facts hard-matched to training:
  * leg_phase is GROUPED (sin of each leg then cos of each leg), exactly as
    `commanded_trot_leg_phase_sin_cos` does torch.cat((sin, cos), dim=1).
    This is NOT the per-leg interleaved [sin_i, cos_i] used by the go2 crawl.
  * diagonal pairs FL+RR (offset 0.0) and FR+RL (offset 0.5), so the two
    diagonal pairs are anti-phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .trot1_contract import TrotGaitConfig

# FL, FR, RL, RR -- diagonal pairs (FL+RR, FR+RL).
TROT_PHASE_OFFSETS = (0.0, 0.5, 0.5, 0.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class TrotGaitTerms:
    velocity_command: tuple[float, float, float]
    global_phase: tuple[float, float]          # (sin, cos) of the global clock
    leg_phase: tuple[float, ...]               # grouped: sin*4 then cos*4
    desired_contacts: tuple[float, float, float, float]
    gait_params: tuple[float, float, float]    # (frequency_hz, duty_factor, swing_height)
    phase: float


class TrotGaitClock:
    """Stateful phase integrator, mirroring the sim's cached clock."""

    def __init__(self, config: TrotGaitConfig, policy_dt: float):
        self.config = config
        self.policy_dt = float(policy_dt)
        self.phase = 0.0
        self.was_moving = False

    def reset(self) -> None:
        self.phase = 0.0
        self.was_moving = False

    def step(self, command: Sequence[float]) -> TrotGaitTerms:
        cfg = self.config
        vx, vy, yaw = float(command[0]), float(command[1]), float(command[2])

        # --- speed_conditioned_gait_parameters (scalar, identical to sim) ---
        moving = math.sqrt(vx * vx + vy * vy + yaw * yaw) >= cfg.command_deadband
        equivalent_speed = math.sqrt(vx * vx + vy * vy) + cfg.yaw_equivalent_radius * abs(yaw)
        blend = _clamp(
            (equivalent_speed - cfg.min_equivalent_speed)
            / max(cfg.max_equivalent_speed - cfg.min_equivalent_speed, 1.0e-6),
            0.0,
            1.0,
        )
        frequency = cfg.min_frequency_hz + blend * (cfg.max_frequency_hz - cfg.min_frequency_hz)
        duty = cfg.low_speed_duty_factor + blend * (cfg.high_speed_duty_factor - cfg.low_speed_duty_factor)
        if not moving:
            frequency, duty = 0.0, 1.0
        height = cfg.swing_height if moving else 0.0

        # --- integrate phase (stateful clock, not a function of elapsed_s) ---
        self.phase = (self.phase + self.policy_dt * frequency) % 1.0
        just_started = moving and not self.was_moving
        if just_started:
            self.phase = _clamp(1.0 - duty, 0.0, 0.5)
        self.was_moving = moving

        swing_fraction = _clamp(1.0 - duty, 0.0, 0.5)
        leg_phases = [(self.phase - offset) % 1.0 for offset in TROT_PHASE_OFFSETS]

        if moving:
            global_phase = (math.sin(2.0 * math.pi * self.phase), math.cos(2.0 * math.pi * self.phase))
            sin_block = [math.sin(2.0 * math.pi * lp) for lp in leg_phases]
            cos_block = [math.cos(2.0 * math.pi * lp) for lp in leg_phases]
            cl_phase = tuple(0.0 if lp < swing_fraction else 1.0 for lp in leg_phases)
            reported_phase = self.phase
        else:
            global_phase = (0.0, 1.0)
            sin_block = [0.0] * 4
            cos_block = [1.0] * 4
            cl_phase = (1.0, 1.0, 1.0, 1.0)
            reported_phase = 0.0

        return TrotGaitTerms(
            velocity_command=(vx, vy, yaw),
            global_phase=global_phase,
            leg_phase=tuple(sin_block + cos_block),  # grouped layout (trot1)
            desired_contacts=cl_phase,
            gait_params=(frequency, duty, height),
            phase=reported_phase,
        )
