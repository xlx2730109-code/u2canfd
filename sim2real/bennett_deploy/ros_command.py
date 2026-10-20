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

    def __init__(self, axes=(3, 4, 0), scales=(0.25, 0.12, 0.5), deadzone=0.05):
        if len(axes) != 3 or any(type(i) is not int or i < 0 for i in axes):
            raise ValueError("expected three nonnegative axis indices")
        if not math.isfinite(deadzone) or not 0 <= deadzone < 1:
            raise ValueError("deadzone must be between zero and one")
        self.axes, self.scales, self.deadzone = tuple(axes), velocity(scales), deadzone

    def command(self, axes: Sequence[float]) -> tuple[float, float, float]:
        try:
            values = velocity(tuple(axes[i] for i in self.axes))
        except (IndexError, TypeError, ValueError, OverflowError):
            return ZERO
        def normalized(value):
            magnitude = max(0.0, min(1.0, abs(value)) - self.deadzone) / (1.0 - self.deadzone)
            return math.copysign(magnitude, value)
        return tuple(normalized(value) * scale for value, scale in zip(values, self.scales))


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


class TeleopGate:
    """Require fresh Joy, a held button, and a command after button engagement."""

    def __init__(self, timeout_s: float = 0.25, deadman_button: int = 4, stop_button: int = 2):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout must be finite and positive")
        if deadman_button < 0 or stop_button < 0 or deadman_button == stop_button:
            raise ValueError("button indices must be distinct and nonnegative")
        self.timeout_s = timeout_s
        self.deadman_button, self.stop_button = deadman_button, stop_button
        self.command = ZERO
        self.command_at = self.joy_at = -math.inf
        self.enabled_since = math.inf
        self.held = False
        self.stop_requested = False

    def update_command(self, values: Sequence[float], now: float | None = None) -> None:
        try:
            self.command = velocity(values)
        except (TypeError, ValueError, OverflowError):
            self.command, self.command_at = ZERO, -math.inf
            return
        self.command_at = time.perf_counter() if now is None else now

    def update_joy(self, buttons: Sequence[int], now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        held = len(buttons) > self.deadman_button and buttons[self.deadman_button] == 1
        if held and (not self.held or now - self.joy_at > self.timeout_s):
            self.enabled_since = now
        if not held:
            self.enabled_since = math.inf
        self.held, self.joy_at = held, now
        if len(buttons) > self.stop_button and buttons[self.stop_button] == 1:
            self.stop_requested = True

    def packet(self, now: float | None = None) -> bytes:
        now = time.perf_counter() if now is None else now
        enabled = (
            self.held and not self.stop_requested
            and 0 <= now - self.joy_at <= self.timeout_s
            and 0 <= now - self.command_at <= self.timeout_s
            and self.command_at >= self.enabled_since
        )
        return json.dumps({
            "v": 1, "sent": now, "enabled": enabled,
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
        self.last_sent = -math.inf
        self.command = ZERO
        self.enabled = self.stop_requested = False
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
            if not isinstance(packet, dict) or packet.get("v") != 1:
                return False
            sent = packet["sent"]
            if isinstance(sent, bool) or not isinstance(sent, (int, float)) or not math.isfinite(sent):
                return False
            if sent <= self.last_sent or not -0.05 <= now - sent <= self.timeout_s:
                return False
            if type(packet["enabled"]) is not bool or type(packet["stop"]) is not bool:
                return False
            values = velocity(packet["command"])
            self.command = tuple(max(lo, min(hi, v)) for v, lo, hi in zip(values, self.minimum, self.maximum))
            self.last_sent, self.enabled = sent, packet["enabled"]
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
        enabled = fresh and self.enabled and not self.stop_requested
        return CommandState(self.command if enabled else ZERO, enabled, fresh, self.stop_requested)

    def close(self) -> None:
        self.socket.close()


class RosArmKeyboard(KeyboardCommandVelocity):
    """Keyboard only arms/exits in ROS mode; it never overwrites ROS velocity."""

    def _apply_key(self, key: str, *, extended: bool = False) -> None:
        if not extended and key.lower() in (" ", "\r", "q", "\x1b"):
            super()._apply_key(key, extended=False)
