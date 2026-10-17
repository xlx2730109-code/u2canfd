from __future__ import annotations

import atexit
import os
import sys
import threading
from typing import List, Tuple


def parse_posix_keys(data: str) -> List[Tuple[str, bool]]:
    """Split raw POSIX terminal input into (key, extended) events.

    The result uses the same key names as the Windows ``msvcrt`` path: PageUp,
    PageDown and Home become the extended keys ``I``, ``Q`` and ``G``; a lone
    ESC is a plain ``\x1b`` (shutdown); other escape sequences (arrows, ...)
    are ignored; ``\n`` is mapped to ``\r`` (arm).
    """

    events: List[Tuple[str, bool]] = []
    index, size = 0, len(data)
    while index < size:
        char = data[index]
        if char == "\x1b":
            if index + 1 < size and data[index + 1] in "[O":
                end = index + 2
                while end < size and not 0x40 <= ord(data[end]) <= 0x7E:
                    end += 1
                sequence = data[index + 2:end + 1]
                index = end + 1
                mapped = {"5~": "I", "6~": "Q", "H": "G", "1~": "G"}.get(sequence)
                if mapped is not None:
                    events.append((mapped, True))
                continue
            events.append(("\x1b", False))
            index += 1
            continue
        events.append(("\r" if char == "\n" else char, False))
        index += 1
    return events


class KeyboardCommandVelocity:
    """Non-blocking keyboard velocity command and policy arm control (Windows msvcrt, POSIX termios)."""

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
        self._posix_fd = None
        self._posix_saved = None
        if os.name == "posix":
            self._setup_posix_terminal()

    def _setup_posix_terminal(self) -> None:
        """Put stdin in cbreak mode (no echo, no line buffering); Ctrl-C still works."""
        if not sys.stdin.isatty():
            print("[KEYBOARD] stdin is not a terminal; keyboard commands are disabled", file=sys.stderr)
            return
        import termios
        import tty

        fd = sys.stdin.fileno()
        self._posix_saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._posix_fd = fd
        atexit.register(self.close)

    def close(self) -> None:
        """Restore the terminal settings changed by the POSIX keyboard path."""
        if self._posix_fd is not None and self._posix_saved is not None:
            import termios

            termios.tcsetattr(self._posix_fd, termios.TCSADRAIN, self._posix_saved)
        self._posix_fd = None
        self._posix_saved = None

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

    def _poll_posix(self) -> None:
        if self._posix_fd is None:
            return
        import select

        data = ""
        try:
            while select.select([self._posix_fd], [], [], 0)[0]:
                chunk = os.read(self._posix_fd, 64)
                if not chunk:
                    break
                data += chunk.decode("utf-8", "ignore")
        except Exception as exc:
            print(f"[KEYBOARD] polling disabled after error: {exc}", file=sys.stderr)
            self._posix_fd = None
            return
        for key, extended in parse_posix_keys(data):
            self._apply_key(key, extended=extended)

    def poll(self) -> Tuple[float, float, float]:
        if sys.platform != "win32":
            self._poll_posix()
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
