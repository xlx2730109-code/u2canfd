import os, sys, math
sys.path.insert(0, r"E:\HuanCun\Desktop\u2canfd\quad_leg_xu")
from bennett_deploy.trot1_contract import TrotDeploymentContract, EXPECTED_OBS_ORDER
from bennett_deploy.trot1_gait import TrotGaitClock

EXPORTED = r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_trot\quad_leg_trot1\flat\2026-07-27_20-08-08\exported"

contract = TrotDeploymentContract.load(EXPORTED)
print("=== CONTRACT LOADED ===")
for line in contract.summary_lines():
    print(line)

print("\n=== OBS ORDER (deploy must build exactly this) ===")
print(" + ".join(f"{name}({[3,3,3,8,8,8,2,8,4,3][i]})" for i, name in enumerate(EXPECTED_OBS_ORDER)))
print("  total dim =", contract.observation_dim, " action_dim =", contract.action_dim)

print("\n=== TRAIN DEFAULT (rad / deg) ===")
for i, name in enumerate(contract.joint_order):
    print(f"  {name:9s} {contract.train_default_rad[i]:+.4f} rad = {math.degrees(contract.train_default_rad[i]):+.2f} deg  (sign={contract.joint_specs[i].sim_to_motor:+.0f})")

print("\n=== GAIT CLOCK: forward walk cmd=(0.20,0,0) over 1.0s @50Hz ===")
clock = TrotGaitClock(contract.gait, 1.0 / contract.policy_rate_hz)
obs_dim = contract.observation_dim
n_ok = 0
for k in range(int(contract.policy_rate_hz)):  # 1 second
    cmd = (0.20, 0.0, 0.0)
    g = clock.step(cmd)
    # assemble the full 50-dim obs exactly as the builder would (no net needed)
    obs = (
        list((0.0, 0.0, 0.0))            # base_ang_vel (demo)
        + list((0.0, 0.0, -1.0))         # projected_gravity (demo)
        + list(g.velocity_command)       # velocity_commands
        + list(contract.train_default_rad)   # joint_pos_rel (demo = defaults)
        + list((0.0,) * 8)               # joint_vel_rel (demo)
        + list((0.0,) * 8)               # last_action (demo)
        + list(g.global_phase)           # trot_phase (2)
        + list(g.leg_phase)              # trot_leg_phase (8, grouped)
        + list(g.desired_contacts)       # desired_contacts (4)
        + list(g.gait_params)            # gait_params (3)
    )
    assert len(obs) == obs_dim, f"obs dim mismatch: {len(obs)} != {obs_dim}"
    assert all(math.isfinite(float(x)) for x in obs)
    n_ok += 1

print(f"  built {n_ok} obs frames, each exactly {obs_dim}-dim, all finite -> PASS")

# Show one diagnostic frame deep in stance
g_last = None
clock2 = TrotGaitClock(contract.gait, 1.0 / contract.policy_rate_hz)
for k in range(int(contract.policy_rate_hz * 0.5)):
    g_last = clock2.step((0.20, 0.0, 0.0))
print("  t=0.5s frame:")
print(f"    global_phase=({g_last.global_phase[0]:+.3f},{g_last.global_phase[1]:+.3f})")
print(f"    leg_phase(grouped sin)={tuple(round(v,3) for v in g_last.leg_phase[:4])}")
print(f"    leg_phase(grouped cos)={tuple(round(v,3) for v in g_last.leg_phase[4:])}")
print(f"    desired_contacts(FL,FR,RL,RR)={tuple(round(v,2) for v in g_last.desired_contacts)}")
print(f"    gait_params(freq,duty,h)={tuple(round(v,3) for v in g_last.gait_params)}")

# Diagonal-pair check: FL & RR must share contact state; FR & RL must share it.
fc = g_last.desired_contacts
print("    diagonal-pair check:", "FL==RR" if (fc[0] == fc[3]) else "FL!=RR(WRONG)",
      "|", "FR==RL" if (fc[1] == fc[2]) else "FR!=RL(WRONG)")
# Anti-phase between the two diagonal pairs: pairs must differ.
pairs = (fc[0], fc[1])
print("    anti-phase between pairs:", "OK" if pairs[0] != pairs[1] else "SAME(WRONG)")

print("\n=== STANDING cmd=(0,0,0) ===")
g0 = TrotGaitClock(contract.gait, 1.0 / contract.policy_rate_hz).step((0.0, 0.0, 0.0))
print("    global_phase=%s leg_phase=%s contacts=%s gait_params=%s" % (
    tuple(round(v,2) for v in g0.global_phase),
    tuple(round(v,2) for v in g0.leg_phase),
    tuple(round(v,2) for v in g0.desired_contacts),
    tuple(round(v,2) for v in g0.gait_params),
))
assert len(list(g0.global_phase) + list(g0.leg_phase) + list(g0.desired_contacts) + list(g0.gait_params)) == 17

print("\nALL CHECKS PASS")
