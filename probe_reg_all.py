# What are reg1 / reg4 on the DM-IMU? Capture their 3-float triples for a few seconds.
from __future__ import annotations
import os, sys, time
sys.path.insert(0, r"C:\Users\xu\AppData\Roaming\Python\Python313\site-packages")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "quad_leg_xu", "bennett_deploy"))
from imu import DMImuSerialReader  # noqa: E402

reader = DMImuSerialReader("COM6", 921600, exit_setting_mode=True)
if not reader.wait_ready(2.0):
    print("not ready", reader.diagnostics())
    raise SystemExit(1)
print("capturing 5s ...", file=sys.stderr)
t0 = time.time()
acc = []
q4 = []
while time.time() - t0 < 5.0:
    for reg in (1, 4):
        v, _ = reader.get_with_age(reg)
        if v is None:
            continue
        if reg == 1:
            acc.append(v)
        else:
            q4.append(v)
    time.sleep(0.002)

def stats(tag, rows):
    if not rows:
        print(f"  reg/{tag}: no samples")
        return
    n = len(rows)
    mean = tuple(sum(r[i] for r in rows) / n for i in range(3))
    mag = (mean[0] ** 2 + mean[1] ** 2 + mean[2] ** 2) ** 0.5
    print(f"  reg/{tag}: n={n} mean=({mean[0]:+.4f},{mean[1]:+.4f},{mean[2]:+.4f}) |mean|={mag:.4f}")

print("REGISTER MEANS (level/current pose):")
stats("1", acc)   # hypothesis: raw accelerometer (g or m/s^2)
stats("4", q4)    # hypothesis: quaternion or temperature
print("diagnostics:", reader.diagnostics())
