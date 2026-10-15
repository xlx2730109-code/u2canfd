"""第 2 关（质心 / 惯量）：从 SW 装配体的叶子零件合成每个 URDF link 的质量、质心、惯量，并与 URDF 对比。

只读：只读零件的质量属性和部件变换，不选择、不保存、不修改任何文档。仅 Windows（SolidWorks COM）。
先用整机做自检：合成结果必须与 SW 自己给出的整机质量属性一致，否则不输出 link 结果。
用法: python sw_link_inertia.py [输出csv路径]
"""
import csv
import sys
from pathlib import Path

if sys.platform != "win32":
    sys.exit("SolidWorks COM 只能在 Windows 上运行")

import numpy as np  # noqa: E402
import win32com.client as w  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[0] / "results" / "r03" / "sw_vs_urdf_link_inertia.csv"
URDF = Path("E:/Project/Isaaclab/bennett_rl/source/bennett_rl/bennett_rl/assets/robots/Urdf_Bennett_3/urdf/Urdf_Bennett_3.urdf")

LEG_OF = {"腿左前": "FL", "腿右前": "FR", "腿左后": "RL", "腿右后": "RR"}
LINK_OF = {"thigh": "thigh", "calf": "calf", "1": "1", "2": "2"}  # 子装配 X_thigh/X_calf/X_1/X_2；足 -> foot；X_base -> base

sw = w.GetActiveObject("SldWorks.Application")
asm = sw.ActiveDoc
if asm is None or not str(asm.GetPathName).upper().endswith(".SLDASM"):
    sys.exit("当前活动文档不是装配体")


TRANSPOSE = False  # 自检时在 True/False 之间选择（ArrayData 的旋转矩阵行/列排布不确定）


def mat4(arr):
    """SW MathTransform.ArrayData -> (R 3x3, t 3)。ArrayData: 9 个旋转 + 3 个平移 + 缩放。"""
    a = np.array(arr, float)
    R = a[:9].reshape(3, 3)
    if TRANSPOSE:
        R = R.T
    t = a[9:12]
    return R, t


def comp_transform(c):
    return mat4(c.Transform2.ArrayData)


def world(c, relative_to_parent):
    """部件在装配体坐标系下的变换。relative_to_parent=True：Transform2 相对父装配体，需逐级相乘。"""
    R, t = comp_transform(c)
    if not relative_to_parent:
        return R, t
    p = c.GetParent
    while p is not None:
        Rp, tp = comp_transform(p)
        t = Rp @ t + tp
        R = Rp @ R
        p = p.GetParent
    return R, t


root = asm.ConfigurationManager.ActiveConfiguration.GetRootComponent3(True)
leaves = []  # (chain names, comp, part-doc mass props)


def walk(c, chain):
    for ch in c.GetChildren:
        name = str(ch.Name2).split("/")[-1]
        path = str(ch.GetPathName)
        if path.upper().endswith(".SLDPRT"):
            if ch.GetSuppression == 0:
                continue
            doc = ch.GetModelDoc2
            if doc is None:
                continue
            mp = doc.GetMassProperties
            if not mp or mp[5] <= 0:
                continue
            leaves.append((chain + [name], ch, mp))
        else:
            walk(ch, chain + [name])


walk(root, [])
print("leaf parts:", len(leaves))


def combine(items):
    """items: [(mass, com(3), I(3x3) about own COM in assembly axes)] -> mass, com, I about combined COM."""
    M = sum(m for m, _, _ in items)
    com = sum(m * c for m, c, _ in items) / M
    I = np.zeros((3, 3))
    for m, c, Ii in items:
        d = c - com
        I += Ii + m * (d @ d * np.eye(3) - np.outer(d, d))
    return M, com, I


def tensor_from_mp(mp, sign):
    Lxx, Lyy, Lzz, Lxy, Lzx, Lyz = mp[6], mp[7], mp[8], mp[9], mp[10], mp[11]
    s = sign
    return np.array([[Lxx, s * Lxy, s * Lzx], [s * Lxy, Lyy, s * Lyz], [s * Lzx, s * Lyz, Lzz]])


def leaf_items(rel, sign, selector=lambda chain: True):
    items = []
    for chain, c, mp in leaves:
        if not selector(chain):
            continue
        R, t = world(c, rel)
        com = R @ np.array(mp[0:3]) + t
        I = R @ tensor_from_mp(mp, sign) @ R.T
        items.append((mp[5], com, I))
    return items


# ---- 自检：整机 ----
tot = asm.GetMassProperties
Iasm = tensor_from_mp(tot, 1)
best = None
for tr in (False, True):
    TRANSPOSE = tr
    for rel in (True, False):
        for sign in (1, -1):
            M, com, I = combine(leaf_items(rel, sign))
            err = np.linalg.norm(I - tensor_from_mp(tot, 1)) / np.linalg.norm(Iasm)
            print("  试 转置=%s parent相对=%s 符号=%+d -> 质心差 %.3f mm, 惯量差 %.3f%%"
                  % (tr, rel, sign, np.linalg.norm(com - np.array(tot[0:3])) * 1000, err * 100))
            if best is None or err < best[0]:
                best = (err, tr, rel, sign, M, com, I)
err, TRANSPOSE, REL, SIGN, M, com, I = best
print("自检（整机）  合成质量 %.5f / SW %.5f | 质心差 %.3f mm | 惯量相对差 %.4f%%  (转置=%s, parent相对=%s, 乘积符号=%+d)"
      % (M, tot[5], np.linalg.norm(com - np.array(tot[0:3])) * 1000, err * 100, TRANSPOSE, REL, SIGN))
if err > 0.01 or abs(M - tot[5]) > 0.01:
    sys.exit("自检未通过（惯量相对差 > 1% 或质量差 > 0.01 kg），不输出 link 结果")

# ---- URDF 侧：link 坐标系 = 关节原点逐级平移（rpy 全 0），质心 = 坐标系原点 + 局部质心 ----
tree = ET.parse(URDF).getroot()
parent_of, origin_of = {}, {}
for j in tree.iter("joint"):
    c = j.find("child").get("link")
    parent_of[c] = j.find("parent").get("link")
    o = j.find("origin")
    origin_of[c] = np.array([float(v) for v in o.get("xyz").split()]) if o is not None else np.zeros(3)
    rpy = o.get("rpy", "0 0 0") if o is not None else "0 0 0"
    assert max(abs(float(v)) for v in rpy.split()) < 1e-9, "关节有旋转，本脚本假设 rpy 全 0"


def link_origin(name):
    t = np.zeros(3)
    while name in parent_of:
        t = t + origin_of[name]
        name = parent_of[name]
    return t


def urdf_link(name):
    l = [x for x in tree.findall("link") if x.get("name") == name][0]
    i = l.find("inertial")
    m = float(i.find("mass").get("value"))
    com = np.array([float(v) for v in i.find("origin").get("xyz").split()]) + link_origin(name)
    a = i.find("inertia").attrib
    I = np.array([[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]], [a["ixz"], a["iyz"], a["izz"]]], float)
    return m, com, I


def classify(chain):
    """叶子零件所属的 URDF link 名。"""
    top = chain[0].rsplit("-", 1)[0]
    if top not in LEG_OF:
        return "base"
    leg = LEG_OF[top]
    sub = chain[1].rsplit("-", 1)[0] if len(chain) > 1 else ""
    if sub.startswith("足"):
        return leg + "_foot"
    if sub.endswith("_base"):
        return "base"
    key = sub.split("_", 1)[-1]
    return leg + "_" + LINK_OF.get(key, "?")


rows = []
names = ["base"] + [f"{l}_{k}" for k in ("thigh", "calf", "1", "2", "foot") for l in ("FL", "FR", "RL", "RR")]
print("\nlink | 质量 SW/URDF | 质心差 mm | 主惯量相对差 %（小/中/大） | 完整张量相对差 %")
for n in names:
    items = leaf_items(REL, SIGN, lambda ch, n=n: classify(ch) == n)
    if not items:
        print(n, "没有叶子零件，跳过")
        continue
    Ms, cs, Is = combine(items)
    Mu, cu, Iu = urdf_link(n)
    ev_s, ev_u = np.linalg.eigvalsh(Is), np.linalg.eigvalsh(Iu)
    rel = (ev_u - ev_s) / ev_s * 100
    fro = np.linalg.norm(Iu - Is) / np.linalg.norm(Is) * 100
    dcom = np.linalg.norm(cu - cs) * 1000
    rows.append([n, Ms, Mu, dcom, *rel, fro, *ev_s, *ev_u])
    print(f"{n:9s} {Ms:.5f}/{Mu:.5f} | {dcom:7.3f} | {rel[0]:+7.2f} {rel[1]:+7.2f} {rel[2]:+7.2f} | {fro:6.2f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.writer(f)
    wr.writerow(["link", "mass_sw", "mass_urdf", "com_diff_mm", "pi_rel_min_pct", "pi_rel_mid_pct", "pi_rel_max_pct",
                 "tensor_rel_pct", "pi_sw_min", "pi_sw_mid", "pi_sw_max", "pi_urdf_min", "pi_urdf_mid", "pi_urdf_max"])
    wr.writerows(rows)
print("saved", OUT)
