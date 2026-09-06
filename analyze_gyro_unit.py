# Non-circular gyro unit check: reg2 gyro should equal d(reg3 euler)/dt.
# Use the pitch ramp in imu_tilt.csv. Compare d(pitch_deg)/dt (deg/s) to wy (reg2).
# If wy tracks d(pitch)/dt in the same numeric scale -> reg2 is deg/s.
# If wy is ~0.017x as large -> reg2 is rad/s.
import csv, math
rows = []
with open(r"E:\HuanCun\Desktop\u2canfd\sysid_out\imu_tilt.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)

print(f"total rows={len(rows)}")
# Column sanity
print("sample:", {k: rows[-1][k] for k in rows[-1]})

def dt_pair(i):
    a, b = rows[i - 1], rows[i]
    t0, t1 = float(a["t"]), float(b["t"])
    return t0, t1

# Scan for the ramp window: d(pitch)/dt largest (the 3s tilt). Then compare peak wy.
print("\n--- ramp window (|dPitch/dt| top 12 samples) ---")
cands = []
for i in range(1, len(rows)):
    t0, t1 = dt_pair(i)
    if t1 <= t0:
        continue
    dpitch = float(rows[i]["pitch_deg"]) - float(rows[i - 1]["pitch_deg"])
    rate = dpitch / (t1 - t0)
    wy = float(rows[i]["wy"])
    cands.append((abs(rate), rate, wy, t1, dpitch, t1 - t0))
cands.sort(reverse=True)
for abs_r, rate, wy, t, dp, dt in cands[:12]:
    # gyro in deg/s vs the Euler-derivative rate (deg/s)
    print(f"  t={t:6.2f} dpitch={dp:+6.3f}dt={dt:.3f} dPitch/dt={rate:+7.2f}deg/s | wy(reg2)={wy:+7.3f} | ratio wy/rate={wy/rate:+8.4f}" if abs(rate) > 1e-6 else f"  t={t:6.2f} dpitch={dp:+6.3f}dt={dt:.3f} dPitch/dt={rate:+7.2f}deg/s | wy(reg2)={wy:+7.3f} | (rate~0)")
