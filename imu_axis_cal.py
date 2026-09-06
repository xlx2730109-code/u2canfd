# IMU axis calibration recorder (COM6). Logs reg3 euler + reg2 gyro + projected_gravity_from_rpy.
# Usage:
#   python imu_axis_cal.py --duration 5            # stationary level sanity (offline motion)
#   python imu_axis_cal.py --duration 40 --out raw.csv   # capture a full tilt sequence
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, r"C:\Users\xu\AppData\Roaming\Python\Python313\site-packages")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quad_leg_xu", "bennett_deploy"))
from imu import DMImuSerialReader, projected_gravity_from_rpy_deg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--baudrate", type=int, default=921600)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--out", default=None, help="CSV path; if None, prints a live console stream")
    args = ap.parse_args()

    reader = DMImuSerialReader(args.port, args.baudrate, exit_setting_mode=True)
    if not reader.wait_ready(2.0):
        print(f"[IMU] not ready on {args.port}: {reader.diagnostics()}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[IMU] ready on {args.port}, recording {args.duration:.1f}s ...", file=sys.stderr)

    t0 = time.time()
    out_handle = None
    writer = None
    if args.out:
        out_handle = open(args.out, "w", newline="", encoding="utf-8")
        writer = csv.writer(out_handle)
        writer.writerow(["t", "roll_deg", "pitch_deg", "yaw_deg", "wx", "wy", "wz", "gx", "gy", "gz"])

    rows = []
    last_tick = -1
    while time.time() - t0 < args.duration:
        rpy, _ = reader.get_with_age(3)
        gyro, _ = reader.get_with_age(2)
        if rpy is None or gyro is None:
            time.sleep(0.002)
            continue
        roll, pitch, yaw = rpy
        g = projected_gravity_from_rpy_deg(roll, pitch, yaw)
        row = [time.time() - t0, roll, pitch, yaw, gyro[0], gyro[1], gyro[2], g[0], g[1], g[2]]
        rows.append(row)
        now_s = time.time() - t0
        if int(now_s) != last_tick:
            last_tick = int(now_s)
            print(f"  [{now_s:5.1f}s] rpy=({roll:+7.2f},{pitch:+7.2f},{yaw:+7.2f}) "
                  f"g=({g[0]:+.4f},{g[1]:+.4f},{g[2]:+.4f}) gtick={int(now_s)}", file=sys.stderr)
        if writer is not None:
            writer.writerow(row)
        time.sleep(0.005)  # ~200Hz

    if out_handle is not None:
        out_handle.close()
        print(f"[IMU] wrote {len(rows)} rows -> {args.out}", file=sys.stderr)

    # Level stats over the final 3.0s (assume stationary there)
    tail = [row for row in rows if row[0] >= max(0.0, args.duration - 3.0)]
    if tail:
        gx = sum(r[7] for r in tail) / len(tail)
        gy = sum(r[8] for r in tail) / len(tail)
        gz = sum(r[9] for r in tail) / len(tail)
        roll_m = sum(r[1] for r in tail) / len(tail)
        pitch_m = sum(r[2] for r in tail) / len(tail)
        roll_s = (sum((r[1] - roll_m) ** 2 for r in tail) / len(tail)) ** 0.5
        pitch_s = (sum((r[2] - pitch_m) ** 2 for r in tail) / len(tail)) ** 0.5
        mag = (gx * gx + gy * gy + gz * gz) ** 0.5
        print("\n[LEVEL] tail-3s: roll=%+.3f deg (sd %.3f) pitch=%+.3f deg (sd %.3f)"
              % (roll_m, roll_s, pitch_m, pitch_s))
        print("[LEVEL] proj_gravity=(%+.4f, %+.4f, %+.4f) |g|=%.4f" % (gx, gy, gz, mag))
        print("[LEVEL] -> |g| should be ~1.0; gz should be ~ -1.0; gx,gy ~ 0 when level.")

    try:
        import os as _os
        _os._exit(0)
    except Exception:
        pass


if __name__ == "__main__":
    main()
