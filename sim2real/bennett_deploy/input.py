from __future__ import annotations

import sys
import threading
from typing import Tuple


class KeyboardCommandVelocity:
    """Non-blocking Windows keyboard velocity command and policy arm control."""

    def __init__(
        self,
        *,
        stop_event: threading.Event,
        initial_x: float,
        initial_y: float,
        initial_yaw: float,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        min_yaw: float,
        max_yaw: float,
        step_x: float,
        step_y: float,
        step_yaw: float,
        min_nonzero_x: float = 0.0,
        auto_arm: bool = False,
    ):
        self.stop_event = stop_event
        self.command_x = float(initial_x)
        self.command_y = float(initial_y)
        self.command_yaw = float(initial_yaw)
        self.min_x, self.max_x = float(min_x), float(max_x)
        self.min_y, self.max_y = float(min_y), float(max_y)
        self.min_yaw, self.max_yaw = float(min_yaw), float(max_yaw)
        self.step_x, self.step_y, self.step_yaw = float(step_x), float(step_y), float(step_yaw)
        self.min_nonzero_x = float(min_nonzero_x)
        self.armed = bool(auto_arm)
        if self.min_x > self.max_x or self.min_y > self.max_y or self.min_yaw > self.max_yaw:
            raise ValueError("keyboard command minimum must not exceed maximum")
        if self.step_x <= 0.0 or self.step_y <= 0.0 or self.step_yaw <= 0.0:
            raise ValueError("keyboard command steps must be > 0")
        if self.min_nonzero_x < 0.0:
            raise ValueError("min_nonzero_x must be >= 0")
        if self.min_nonzero_x > 0.0 and (
            self.min_x > -self.min_nonzero_x or self.max_x < self.min_nonzero_x
        ):
            raise ValueError("x command range must include +/-min_nonzero_x")
        self.command_x = max(self.min_x, min(self.max_x, self.command_x))
        if 0.0 < abs(self.command_x) < self.min_nonzero_x:
            raise ValueError("initial_x must be zero or have magnitude >= min_nonzero_x")
        self.command_y = max(self.min_y, min(self.max_y, self.command_y))
        self.command_yaw = max(self.min_yaw, min(self.max_yaw, self.command_yaw))

    def _print(self) -> None:
        print(
            f"[KEYBOARD] cmd=[{self.command_x:+.3f}, {self.command_y:+.3f}, {self.command_yaw:+.3f}] "
            "(vx m/s, vy m/s, yaw rad/s)"
        )

    def _zero(self) -> None:
        self.command_x = self.command_y = self.command_yaw = 0.0
        self._print()

    def _step_command_x(self, direction: int) -> None:
        """Move on a symmetric x-speed ladder while skipping untrained near-zero commands."""
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        if self.min_nonzero_x <= 0.0:
            self.command_x = max(
                self.min_x,
                min(self.max_x, self.command_x + direction * self.step_x),
            )
            return

        value = self.command_x
        floor = self.min_nonzero_x
        eps = 1.0e-9
        if direction > 0:
            if value < -floor - eps:
                value = min(value + self.step_x, -floor)
            elif value < -eps:
                value = 0.0
            elif value < floor - eps:
                value = floor
            else:
                value += self.step_x
        else:
            if value > floor + eps:
                value = max(value - self.step_x, floor)
            elif value > eps:
                value = 0.0
            elif value > -floor + eps:
                value = -floor
            else:
                value -= self.step_x
        self.command_x = max(self.min_x, min(self.max_x, value))

    def _apply_key(self, key: str, *, extended: bool = False) -> None:
        lower = key.lower() if len(key) == 1 else key
        if extended and key.upper() == "I":
            self._step_command_x(+1)
            self._print()
        elif extended and key.upper() == "Q":
            self._step_command_x(-1)
            self._print()
        elif extended and key.upper() == "G":
            self._zero()
        elif key in ("\r", " "):
            if not self.armed:
                self.armed = True
                print("[KEYBOARD] policy ARMED")
        elif lower in ("0", "x"):
            self._zero()
        elif lower == "w":
            self._step_command_x(+1)
            self._print()
        elif lower == "s":
            self._step_command_x(-1)
            self._print()
        elif lower == "a":
            self.command_y = min(self.max_y, self.command_y + self.step_y)
            self._print()
        elif lower == "d":
            self.command_y = max(self.min_y, self.command_y - self.step_y)
            self._print()
        elif lower == "j":
            self.command_yaw = min(self.max_yaw, self.command_yaw + self.step_yaw)
            self._print()
        elif lower == "l":
            self.command_yaw = max(self.min_yaw, self.command_yaw - self.step_yaw)
            self._print()
        elif lower in ("q", "\x1b"):
            print("[KEYBOARD] shutdown requested")
            self.stop_event.set()

    def poll(self) -> Tuple[float, float, float]:
        if sys.platform != "win32":
            return self.command_x, self.command_y, self.command_yaw
        try:
            import msvcrt

            while msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in ("\x00", "\xe0"):
                    self._apply_key(msvcrt.getwch(), extended=True)
                else:
                    self._apply_key(key)
        except Exception as exc:
            print(f"[KEYBOARD] polling disabled after error: {exc}", file=sys.stderr)
        return self.command_x, self.command_y, self.command_yaw
