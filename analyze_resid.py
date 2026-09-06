import csv, math
rows = []
with open(r"E:\HuanCun\Desktop\u2canfd\sysid_out\imu_tilt.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)
worst = 0.0
worst_row = None
n = 0
for row in rows:
    roll = float(row["roll_deg"])
    pitch = float(row["pitch_deg"])
    sr, cr = math.sin(math.radians(roll)), math.cos(math.radians(roll))
    sp, cp = math.sin(math.radians(pitch)), math.cos(math.radians(pitch))
    pred = (sp, -sr * cp, -cr * cp)
    obs = (float(row["gx"]), float(row["gy"]), float(row["gz"]))
    for a, b in zip(pred, obs):
        d = abs(a - b)
        if d > worst:
            worst = d
            worst_row = (float(row["t"]), roll, pitch, obs, pred)
        n += 1
print(f"samples={n // 3} points, worst |pred-obs| across gx/gy/gz = {worst:.6f}")
if worst_row:
    t, roll, pitch, obs, pred = worst_row
    print(f"  worst at t={t:.2f}s roll={roll:+.3f} pitch={pitch:+.3f}")
    print(f"  obs ={tuple(round(x, 6) for x in obs)}")
    print(f"  pred={tuple(round(x, 6) for x in pred)}")
