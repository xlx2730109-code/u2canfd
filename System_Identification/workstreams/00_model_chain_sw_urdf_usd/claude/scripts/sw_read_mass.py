"""第 1 关：只读 SolidWorks 装配体的质量属性（整机 + 每个零件）。

只在 Windows 上运行（SolidWorks COM）。只读：不保存、不关闭、不修改任何文档。
用法: python sw_read_mass.py [输出csv路径]
"""
import csv
import sys
from collections import Counter
from pathlib import Path

if sys.platform != "win32":
    sys.exit("SolidWorks COM 只能在 Windows 上运行")

import win32com.client as w  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "r01" / "sw_parts_mass.csv"

sw = w.GetActiveObject("SldWorks.Application")
asm = sw.ActiveDoc
if asm is None or not str(asm.GetPathName).upper().endswith(".SLDASM"):
    sys.exit("当前活动文档不是装配体")
print("assembly:", asm.GetPathName)

tot = asm.GetMassProperties  # (cx,cy,cz,vol,area,mass,Lxx,Lyy,Lzz,Lxy,Lzx,Lyz)  m / kg / kg·m²
print("ASSEMBLY mass=%.4f kg  COM(m)=(%.5f, %.5f, %.5f)  I(Lxx,Lyy,Lzz)=(%.4f, %.4f, %.4f)"
      % (tot[5], tot[0], tot[1], tot[2], tot[6], tot[7], tot[8]))

root = asm.ConfigurationManager.ActiveConfiguration.GetRootComponent3(True)
inst = []


def walk(c):
    for ch in c.GetChildren:
        inst.append(ch)
        walk(ch)


walk(root)
print("components (all levels):", len(inst))

count = Counter(str(c.GetPathName) for c in inst)
seen = {}
rows = []
for c in inst:
    path = str(c.GetPathName)
    if path in seen:
        continue
    seen[path] = True
    sup = c.GetSuppression  # 0 suppressed, 1 lightweight, 2 resolved
    doc = c.GetModelDoc2
    mass = vol = None
    ovr = ""
    cx = cy = cz = None
    ix = iy = iz = None
    if doc is not None and path.upper().endswith(".SLDPRT"):
        mp = doc.GetMassProperties
        if mp:
            cx, cy, cz, vol, _, mass, ix, iy, iz = mp[0], mp[1], mp[2], mp[3], mp[4], mp[5], mp[6], mp[7], mp[8]
        try:
            m2 = doc.Extension.CreateMassProperty  # late-bound 下是属性，不加括号
            flags = []
            for name in ("OverrideMass", "OverrideCenterOfMass", "OverrideMomentsOfInertia"):
                try:
                    if getattr(m2, name):
                        flags.append(name)
                except Exception:
                    pass
            ovr = "+".join(flags)
        except Exception:
            ovr = "n/a"
    dens = (mass / vol) if (mass and vol) else None
    rows.append([Path(path).name, count[path], sup, mass, vol, dens, cx, cy, cz, ix, iy, iz, ovr])

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.writer(f)
    wr.writerow(["part", "instances", "suppression", "mass_kg", "volume_m3", "density_kg_m3",
                 "com_x_m", "com_y_m", "com_z_m", "Lxx", "Lyy", "Lzz", "override"])
    wr.writerows(rows)
print("saved", OUT, "parts:", len(rows))
parts_total = sum((r[3] or 0) * r[1] for r in rows)
print("sum(part mass x instances) = %.4f kg (含标准件；与整机质量对比)" % parts_total)
print("overrides:", [(r[0], r[12]) for r in rows if r[12] and r[12] != "n/a"])
print("no-mass (lightweight/assembly):", [r[0] for r in rows if r[3] is None][:20])
