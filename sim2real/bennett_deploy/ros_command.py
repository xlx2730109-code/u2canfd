"""Local ROS-to-policy transport. No ROS, torch, IMU or motor dependency."""

from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import dataclass
from typing import Sequence

from .input import KeyboardCommandVelocity

ZERO = (0.0, 0.0, 0.0)
DEFAULT_PORT = 19761


class JoystickVelocity:
    """Proportional right-stick translation and left-stick yaw (ROS Joy axes)."""

    def __init__(self, axes=(3, 2, 0), scales=(0.25, 0.12, 0.5), deadzone=0.05, centers=ZERO):
        if len(axes) != 3 or any(type(i) is not int or i < 0 for i in axes):
            raise ValueError("expected three nonnegative axis indices")
        if not math.isfinite(deadzone) or not 0 <= deadzone < 1:
            raise ValueError("deadzone must be between zero and one")
        self.axes, self.scales, self.deadzone = tuple(axes), velocity(scales), deadzone
        self.centers = velocity(centers)
        if any(abs(center) >= 1 for center in self.centers):
            raise ValueError("axis centers must be strictly between minus one and one")

    def command(self, axes: Sequence[float]) -> tuple[float, float, float]:
        try:
            values = velocity(tuple(axes[i] for i in self.axes))
        except (IndexError, TypeError, ValueError, OverflowError):
            return ZERO
        def normalized(value, center):
            # Preserve both full-scale endpoints when the measured center is offset.
            delta = max(-1.0, min(1.0, value)) - center
            span = 1.0 - center if delta >= 0 else 1.0 + center
            displacement = delta / span
            magnitude = max(0.0, abs(displacement) - self.deadzone) / (1.0 - self.deadzone)
            return math.copysign(magnitude, displacement)
        return tuple(normalized(value, center) * scale
                     for value, center, scale in zip(values, self.centers, self.scales))


def velocity(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3 or any(isinstance(v, bool) for v in values):
        raise ValueError("expected three numeric velocity values")
    result = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in result):
        raise ValueError("velocity must be finite")
    return result


@dataclass(frozen=True)
class CommandState:
    command: tuple[float, float, float] = ZERO
    enabled: bool = False
    fresh: bool = False
    stop_requested: bool = False
    start_requested: bool = False
    unlocked: bool = False


class TeleopGate:
    """Require fresh Joy, a held button, and a command after button engagement."""

    def __init__(self, timeout_s: float = 0.25, deadman_button: int = 4,
                 pause_button: int = 1, start_button: int = 9, start_hold_s: float = 1.0):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout must be finite and positive")
        indices = (deadman_button, pause_button, start_button)
        if any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != 3:
            raise ValueError("button indices must be distinct and nonnegative")
        if not math.isfinite(start_hold_s) or start_hold_s <= 0:
            raise ValueError("START hold duration must be finite and positive")
        self.timeout_s = timeout_s
        self.deadman_button, self.pause_button, self.start_button = indices
        self.start_hold_s = start_hold_s
        self.command = ZERO
        self.command_at = self.joy_at = -math.inf
        self.enabled_since = math.inf
        self.held = False
        self.stop_requested = False
        self.locked = True
        self.arm_at = 0.0
        self.start_since = None
        self.start_consumed = False

    def update_command(self, values: Sequence[float], now: float | None = None) -> None:
        try:
            self.command = velocity(values)
        except (TypeError, ValueError, OverflowError):
            self.command, self.command_at = ZERO, -math.inf
            return
        self.command_at = time.perf_counter() if now is None else now

    def update_joy(self, buttons: Sequence[int], now: float | None = None, *, neutral: bool = False) -> None:
        now = time.perf_counter() if now is None else now
        if now - self.joy_at > self.timeout_s:
            self.locked = True
            self.start_since = None
        pressed = lambda index: len(buttons) > index and buttons[index] == 1
        held = len(buttons) > self.deadman_button and buttons[self.deadman_button] == 1
        if held and (not self.held or now - self.joy_at > self.timeout_s):
            self.enabled_since = now
        if not held:
            self.enabled_since = math.inf
        self.held, self.joy_at = held, now
        if pressed(self.pause_button):
            self.locked = True
            self.start_since = None
            self.start_consumed = True
        elif not pressed(self.start_button):
            self.start_since = None
            self.start_consumed = False
        elif not neutral or held:
            self.start_since = None
        elif not self.start_consumed:
            if self.start_since is None:
                self.start_since = now
            elif now - self.start_since >= self.start_hold_s:
                self.locked = False
                self.arm_at = now
                self.start_consumed = True

    def packet(self, now: float | None = None) -> bytes:
        now = time.perf_counter() if now is None else now
        joy_fresh = 0 <= now - self.joy_at <= self.timeout_s
        if not joy_fresh:
            self.locked = True
            self.start_since = None
        enabled = (
            self.held and not self.locked and not self.stop_requested
            and joy_fresh
            and 0 <= now - self.command_at <= self.timeout_s
            and self.command_at >= self.enabled_since
        )
        return json.dumps({
            "v": 2, "sent": now, "enabled": enabled,
            "arm": self.arm_at, "unlocked": joy_fresh and not self.locked,
            "stop": self.stop_requested, "command": self.command if enabled else ZERO,
        }, allow_nan=False).encode("ascii")


class RosCommandReceiver:
    """Nonblocking loopback receiver; policy process enforces its own bounds."""

    def __init__(self, *, port: int = DEFAULT_PORT, timeout_s: float = 0.25,
                 minimum: Sequence[float], maximum: Sequence[float]):
        self.minimum, self.maximum = velocity(minimum), velocity(maximum)
        if any(lo > hi or lo > 0 or hi < 0 for lo, hi in zip(self.minimum, self.maximum)):
            raise ValueError("velocity bounds must be ordered and include zero")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout must be finite and positive")
        self.timeout_s = timeout_s
        self.created_at = time.perf_counter()
        self.last_sent = -math.inf
        self.command = ZERO
        self.enabled = self.stop_requested = False
        self.arm_at = 0.0
        self.unlocked = False
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind(("127.0.0.1", port))
            self.socket.setblocking(False)
        except BaseException:
            self.socket.close()
            raise

    def accept(self, data: bytes, now: float) -> bool:
        try:
            packet = json.loads(data)
            if not isinstance(packet, dict) or packet.get("v") != 2:
                return False
            sent = packet["sent"]
            if isinstance(sent, bool) or not isinstance(sent, (int, float)) or not math.isfinite(sent):
                return False
            if sent <= self.last_sent or not -0.05 <= now - sent <= self.timeout_s:
                return False
            if any(type(packet[key]) is not bool for key in ("enabled", "stop", "unlocked")):
                return False
            arm = packet["arm"]
            if isinstance(arm, bool) or not isinstance(arm, (int, float)) or not math.isfinite(arm) or not 0 <= arm <= sent:
                return False
            values = velocity(packet["command"])
            self.command = tuple(max(lo, min(hi, v)) for v, lo, hi in zip(values, self.minimum, self.maximum))
            self.last_sent, self.enabled = sent, packet["enabled"]
            self.arm_at, self.unlocked = arm, packet["unlocked"]
            self.stop_requested = self.stop_requested or packet["stop"]
            return True
        except (KeyError, TypeError, ValueError, OverflowError, UnicodeError):
            return False

    def poll(self) -> CommandState:
        now = time.perf_counter()
        # Bound work in the motor loop even if a local process floods the port.
        for _ in range(64):
            try:
                data, peer = self.socket.recvfrom(1025)
            except BlockingIOError:
                break
            if peer[0] == "127.0.0.1" and len(data) <= 1024:
                self.accept(data, now)
        fresh = -0.05 <= now - self.last_sent <= self.timeout_s
        start_requested = fresh and self.unlocked and self.arm_at >= self.created_at
        enabled = start_requested and self.enabled and not self.stop_requested
        return CommandState(self.command if enabled else ZERO, enabled, fresh, self.stop_requested,
                            start_requested, fresh and self.unlocked)

    def close(self) -> None:
        self.socket.close()


class RosArmKeyboard(KeyboardCommandVelocity):
    """ROS mode starts from the gamepad; the keyboard only requests exit."""

    def _apply_key(self, key: str, *, extended: bool = False) -> None:
        if not extended and key.lower() in ("q", "\x1b"):
            super()._apply_key(key, extended=False)
