"""Windows：直接调用 libdm_device.dll 同款的 std::this_thread::sleep_for（libstdc++ → libwinpthread 的 nanosleep），
看睡不同时长实际多久；并对比 timeBeginPeriod(1) 前后，以及 kernel32.Sleep(1)。结果见根 README 的 01-K37。

用法：python probe_sleep_for.py
"""
import ctypes
import os
import statistics
import sys
import time
from pathlib import Path

if sys.platform != "win32":
    raise SystemExit("仅 Windows")

d = str(Path(__file__).resolve().parents[5] / "sim2real" / "sdk" / "dlls_lowlatency")
os.add_dll_directory(d)
for n in ("libgcc_s_seh-1.dll", "libwinpthread-1.dll", "libstdc++-6.dll"):
    lib = ctypes.CDLL(os.path.join(d, n))
name = "_ZNSt11this_thread11__sleep_forENSt6chrono8durationIxSt5ratioILx1ELx1EEEENS1_IxS2_ILx1ELx1000000000EEEE"
sleep_for = getattr(lib, name)
sleep_for.argtypes = [ctypes.c_longlong, ctypes.c_longlong]
sleep_for.restype = None
winmm = ctypes.WinDLL("winmm")
k32 = ctypes.WinDLL("kernel32")


def measure(fn, n=100):
    t_all = []
    for _ in range(n):
        t = time.perf_counter_ns()
        fn()
        t_all.append((time.perf_counter_ns() - t) / 1e6)
    return round(statistics.median(t_all), 3), round(max(t_all), 3)


def table(tag):
    print(f"-- {tag}")
    print("kernel32.Sleep(1)      中位/最大 ms", measure(lambda: k32.Sleep(1)))
    for ns in (0, 1000, 100_000, 500_000, 1_000_000, 2_000_000, 10_000_000, 16_000_000):
        print(f"sleep_for({ns / 1e6:g} ms)".ljust(23), "中位/最大 ms", measure(lambda: sleep_for(0, ns)))


table("初始")
winmm.timeBeginPeriod(1)
table("timeBeginPeriod(1) 之后")
winmm.timeEndPeriod(1)
