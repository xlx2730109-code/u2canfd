# 与damiao_2.py配合的单腿RL policy UDP发送器





from __future__ import annotations

import argparse
import json
import math
import socket
import time
from pathlib import Path

import torch


DEFAULT_POLICY = (
    # r"D:\IsaacLab\logs\rsl_rl\Bennett_single_leg_rr_trace\2026-06-19_15-43-00\exported\policy.pt"   #50Hz
    # r"D:\IsaacLab\logs\rsl_rl\Bennett_single_leg_rr_trace\2026-06-19_18-45-06\exported\policy.pt"   #250Hz
    r"D:\IsaacLab\logs\rsl_rl\Bennett_single_leg_rr_trace\2026-06-19_21-23-57\exported\policy.pt"   #100Hz
    # r"D:\IsaacLab\logs\rsl_rl\Bennett_single_leg_rr_trace\2026-06-19_21-49-38\exported\policy.pt"   #150Hz

)


def build_reference(elapsed_s: float, amplitude_rad: float, max_speed_rad_s: float) -> tuple[float, float, float, float]:
    """Return phase sin/cos and thigh/calf reference offsets."""
    frequency_hz = max_speed_rad_s / max(2.0 * math.pi * amplitude_rad, 1.0e-6)
    phase = 2.0 * math.pi * frequency_hz * elapsed_s
    phase_sin = math.sin(phase)
    phase_cos = math.cos(phase)
    thigh_ref = amplitude_rad * phase_sin
    calf_ref = amplitude_rad * math.sin(phase + math.pi / 2.0)
    return phase_sin, phase_cos, thigh_ref, calf_ref


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bennett single-leg policy and stream targets to damiao_2.py.")
    parser.add_argument("--policy", type=str, default=DEFAULT_POLICY, help="Exported TorchScript policy.pt path.")
    parser.add_argument("--udp_target", type=str, default="127.0.0.1:15001", help="UDP target host:port.")
    parser.add_argument("--rate_hz", type=float, default=50.0, help="Policy update and UDP send rate.")
    parser.add_argument("--amplitude_deg", type=float, default=20.0, help="Reference trajectory amplitude.")
    # old: default=35.0，沿用手动键盘测试速度；较大输出幅度时 target 会明显落后 desired。
    # 新：默认放宽到 80deg/s；仍可用 --speed_deg_s 手动覆盖。
    parser.add_argument("--speed_deg_s", type=float, default=80.0, help="Reference max joint speed.")
    parser.add_argument(
        "--output_scale",
        type=float,
        default=0.10,
        help="Extra safety scale for policy output. Use 0.10 first, then raise slowly.",
    )
    # old: default=20.0，20 秒后自动结束；真机端会因为 UDP 断流停止。
    # 新：默认 0 表示一直运行，手动 Ctrl+C 停止。
    parser.add_argument("--duration_s", type=float, default=0.0, help="Run duration. Use 0 for infinite.")
    parser.add_argument("--warmup_s", type=float, default=2.0, help="Send zero targets before running policy.")
    parser.add_argument("--dry_run", action="store_true", help="Print targets without sending UDP.")
    args = parser.parse_args()

    policy_path = Path(args.policy)
    if not policy_path.exists():
        raise FileNotFoundError(policy_path)

    host, port_text = args.udp_target.rsplit(":", 1)
    udp_target = (host, int(port_text))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()

    dt = 1.0 / args.rate_hz
    action_scale_rad = math.radians(20.0)
    amplitude_rad = math.radians(args.amplitude_deg)
    max_speed_rad_s = math.radians(args.speed_deg_s)
    last_action = torch.zeros(2, dtype=torch.float32)
    joint_pos = torch.zeros(2, dtype=torch.float32)
    prev_joint_pos = torch.zeros(2, dtype=torch.float32)

    print(f"[POLICY] loaded: {policy_path}")
    print(f"[POLICY] udp_target={args.udp_target} rate={args.rate_hz:.1f}Hz output_scale={args.output_scale:.3f}")
    print(f"[POLICY] duration_s={args.duration_s:.1f} warmup_s={args.warmup_s:.1f} speed_limit={args.speed_deg_s:.1f}deg/s")
    print("[POLICY] start damiao_2.py first. Ctrl+C to stop.")

    start = time.monotonic()
    next_time = start
    loop_count = 0
    try:
        while args.duration_s <= 0.0 or time.monotonic() - start < args.duration_s:
            now = time.monotonic()
            if next_time > now:
                time.sleep(next_time - now)
            elapsed_s = time.monotonic() - start
            next_time += dt
            loop_count += 1

            if elapsed_s < args.warmup_s:
                action = torch.zeros(2, dtype=torch.float32)
                target_offset = torch.zeros(2, dtype=torch.float32)
                desired_offset = torch.zeros(2, dtype=torch.float32)
                source = "single_leg_policy_warmup"
                prev_joint_pos = joint_pos.clone()
                joint_pos = target_offset.clone()
            else:
                policy_elapsed_s = elapsed_s - args.warmup_s
                phase_sin, phase_cos, thigh_ref, calf_ref = build_reference(
                    policy_elapsed_s, amplitude_rad, max_speed_rad_s
                )
                ref = torch.tensor([thigh_ref, calf_ref], dtype=torch.float32)
                joint_vel = (joint_pos - prev_joint_pos) / dt
                tracking_error = joint_pos - ref
                obs = torch.cat(
                    (
                        torch.tensor([phase_sin, phase_cos], dtype=torch.float32),
                        ref,
                        tracking_error,
                        joint_pos,
                        joint_vel,
                        last_action,
                    )
                ).unsqueeze(0)

                with torch.no_grad():
                    action = policy(obs).squeeze(0).to(torch.float32).clamp(-1.0, 1.0)

                desired_offset = action * action_scale_rad * float(args.output_scale)
                max_delta = max_speed_rad_s * dt
                delta = torch.clamp(desired_offset - joint_pos, -max_delta, max_delta)
                target_offset = joint_pos + delta

                prev_joint_pos = joint_pos.clone()
                joint_pos = target_offset.clone()
                last_action = action.clone()
                source = "single_leg_policy"

            payload = {
                "leg": "FR",
                "thigh_offset_rad": float(target_offset[0].item()),
                "calf_offset_rad": float(target_offset[1].item()),
                "time_s": float(elapsed_s),
                "source": source,
            }
            if not args.dry_run:
                sock.sendto(json.dumps(payload, separators=(",", ":")).encode("ascii"), udp_target)

            if loop_count % max(1, int(args.rate_hz)) == 0:
                print(
                    f"t={elapsed_s:6.2f}s "
                    f"target_deg=({math.degrees(payload['thigh_offset_rad']):+.2f}, "
                    f"{math.degrees(payload['calf_offset_rad']):+.2f}) "
                    f"desired_deg=({math.degrees(float(desired_offset[0].item())):+.2f}, "
                    f"{math.degrees(float(desired_offset[1].item())):+.2f}) "
                    f"action=({action[0].item():+.3f}, {action[1].item():+.3f})"
                )
    except KeyboardInterrupt:
        print("\n[POLICY] stopped by user")
    finally:
        hold_payload = {
            "leg": "FR",
            "thigh_offset_rad": float(joint_pos[0].item()),
            "calf_offset_rad": float(joint_pos[1].item()),
            "time_s": float(time.monotonic() - start),
            "source": "single_leg_policy_hold",
        }
        if not args.dry_run:
            sock.sendto(json.dumps(hold_payload, separators=(",", ":")).encode("ascii"), udp_target)


if __name__ == "__main__":
    main()
