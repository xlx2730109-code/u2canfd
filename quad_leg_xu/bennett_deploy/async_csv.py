from __future__ import annotations

import csv
import queue
import threading
import time
from pathlib import Path
from typing import Iterable, Sequence


CSV_HEADER = (
    "time_monotonic", "elapsed_s", "loop_count", "policy_step", "startup_phase", "diag_stage",
    "contract_id", "command_x", "command_y", "command_yaw", "gait_phase", "joint_name", "joint_active",
    "kp", "kd", "output_scale", "action_clip", "policy_rate_hz", "target_rate_limit_deg_s",
    "software_effort_limit_nm",
    "raw_action", "action", "desired_offset_rad", "applied_offset_rad", "target_sim_rad", "target_sim_deg",
    "train_default_rad", "rel_rad", "rel_deg", "q_policy_motor_rad", "q_cmd_motor_rad", "q_real_motor_rad",
    "dq_real_motor_rad_s", "dq_policy_obs_rad_s", "tau_nm", "motor_err", "feedback_age_s",
    "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z", "projected_gravity_x", "projected_gravity_y",
    "projected_gravity_z", "imu_gyro_age_s", "imu_rpy_age_s", "desired_contact",
)


class AsyncCsvLogger:
    """Bounded, non-blocking producer with a dedicated disk writer."""

    _STOP = object()

    def __init__(self, path: str | Path, *, queue_batches: int = 512, overwrite: bool = False):
        requested = Path(path).expanduser().resolve()
        self.path = requested
        if self.path.exists() and not overwrite:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            candidate = requested.with_name(f"{requested.stem}-{stamp}{requested.suffix}")
            index = 1
            while candidate.exists():
                candidate = requested.with_name(f"{requested.stem}-{stamp}-{index}{requested.suffix}")
                index += 1
            self.path = candidate
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_batches)))
        self._dropped_batches = 0
        self._written_rows = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="deploy-csv-writer", daemon=True)
        self._thread.start()

    def submit(self, rows: Iterable[Sequence[object]]) -> None:
        batch = tuple(tuple(row) for row in rows)
        if not batch:
            return
        try:
            self._queue.put_nowait(batch)
        except queue.Full:
            self._dropped_batches += 1

    def _run(self) -> None:
        try:
            with self.path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(CSV_HEADER)
                last_flush = time.monotonic()
                while True:
                    item = self._queue.get()
                    if item is self._STOP:
                        break
                    writer.writerows(item)
                    self._written_rows += len(item)
                    now = time.monotonic()
                    if now - last_flush >= 1.0:
                        stream.flush()
                        last_flush = now
                stream.flush()
        except BaseException as exc:
            self._error = exc

    def close(self, timeout_s: float = 5.0) -> dict[str, object]:
        try:
            self._queue.put(self._STOP, timeout=max(0.1, timeout_s / 2.0))
        except queue.Full:
            self._dropped_batches += self._queue.qsize()
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait(self._STOP)
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise RuntimeError("CSV writer did not stop within timeout")
        if self._error is not None:
            raise RuntimeError(f"CSV writer failed: {self._error}") from self._error
        return {
            "path": str(self.path),
            "written_rows": self._written_rows,
            "dropped_batches": self._dropped_batches,
        }
