import csv, math
rows = []
with open(r"E:\HuanCun\Desktop\u2canfd\sysid_out\imu_tilt.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)

sel = [row for row in rows if abs(float(row["pitch_deg"])) > 4.0]
print("rows with |pitch|>4deg:", len(sel))
if not sel:
    raise SystemExit("no significant pitch tilt captured")
for row in sel[::30]:
    print(f"  t={float(row['t']):6.2f} roll={float(row['roll_deg']):+6.2f} pitch={float(row['pitch_deg']):+6.2f} "
          f"w=({float(row['wx']):+7.2f},{float(row['wy']):+7.2f},{float(row['wz']):+7.2f}) "
          f"g=({float(row['gx']):+.4f},{float(row['gy']):+.4f},{float(row['gz']):+.4f})")

peak = max(sel, key=lambda r: abs(float(r["pitch_deg"])))
p = float(peak["pitch_deg"])
print(f"\nPEAK pitch={p:+.2f}deg at t={float(peak['t']):.2f}s -> gx={float(peak['gx']):+.4f}, sin(pitch)={math.sin(math.radians(p)):+.4f}")
print(f"  expected gx=sin(pitch)={math.sin(math.radians(p)):+.4f} vs observed gx={float(peak['gx']):+.4f}")
pivot = max(sel, key=lambda r: abs(float(r["wy"])))
print(f"  max gyro|wy|={float(pivot['wy']):+.3f} deg/s (pivot axis for pitch)")
print(f"  cross-talk at pivot: roll={float(pivot['roll_deg']):+.2f}deg wx={float(pivot['wx']):+.3f} wz={float(pivot['wz']):+.3f}")
