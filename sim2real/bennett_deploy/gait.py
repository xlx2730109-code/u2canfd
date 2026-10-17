from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GaitTerms:
    velocity_command: tuple[float, float, float]
    global_phase: tuple[float, float]
    leg_phase: tuple[float, ...]
    desired_contacts: tuple[float, float, float, float]
    gait_params: tuple[float, float, float]
    phase: float


def crawl_phase_terms(
    elapsed_s: float,
    frequency_hz: float,
    duty_factor: float,
    swing_height: float,
    command: Sequence[float],
    command_deadband: float,
) -> GaitTerms:
    if len(command) != 3:
        raise ValueError("velocity command must contain vx, vy, yaw")
    command_tuple = tuple(float(value) for value in command)
    moving = math.sqrt(sum(value * value for value in command_tuple)) >= float(command_deadband)
    if not moving:
        return GaitTerms(
            velocity_command=(0.0, 0.0, 0.0),
            global_phase=(0.0, 1.0),
            leg_phase=(0.0, 1.0) * 4,
            desired_contacts=(1.0, 1.0, 1.0, 1.0),
            gait_params=(0.0, 1.0, 0.0),
            phase=0.0,
        )

    phase = (float(elapsed_s) * float(frequency_hz)) % 1.0
    phase_rad = 2.0 * math.pi * phase
    leg_phase: list[float] = []
    desired_contacts: list[float] = []
    swing_fraction = max(0.0, 1.0 - float(duty_factor))
    # FL, FR, RL, RR; this order is part of the trained observation contract.
    for offset in (0.0, 0.5, 0.75, 0.25):
        value = (phase - offset) % 1.0
        leg_phase.extend((math.sin(2.0 * math.pi * value), math.cos(2.0 * math.pi * value)))
        desired_contacts.append(0.0 if value < swing_fraction else 1.0)
    return GaitTerms(
        velocity_command=command_tuple,
        global_phase=(math.sin(phase_rad), math.cos(phase_rad)),
        leg_phase=tuple(leg_phase),
        desired_contacts=tuple(desired_contacts),
        gait_params=(float(frequency_hz), float(duty_factor), float(swing_height)),
        phase=phase,
    )
