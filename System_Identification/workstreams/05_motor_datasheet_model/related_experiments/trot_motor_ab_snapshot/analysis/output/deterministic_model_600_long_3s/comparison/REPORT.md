# Trot motor A/B deterministic diagnostic

- A source: `E:\Project\Isaaclab\bennett_rl\source\bennett_rl\bennett_rl\tasks\manager_based\quad_leg_trot\quad_leg_trot-motor-ab\analysis\output\deterministic_model_600_long_3s\a.csv`
- B source: `E:\Project\Isaaclab\bennett_rl\source\bennett_rl\bennett_rl\tasks\manager_based\quad_leg_trot\quad_leg_trot-motor-ab\analysis\output\deterministic_model_600_long_3s\b.csv`
- Common scenarios: backward_nominal, backward_slow, backward_yaw, forward_nominal, forward_slow, forward_yaw, lateral_left, stand, yaw_left, yaw_right
- A rows used: 1120
- B rows used: 1120

The report is descriptive. It does not declare a winner automatically,
because lower is preferable for most errors/loads while swing height and
contact behavior require task-specific interpretation.

| Metric | A mean | B mean | B/A |
|---|---:|---:|---:|
| velocity_x_rmse | 0.0207734 | 0.0243991 | 1.1745 |
| yaw_rate_rmse | 0.0559018 | 0.0502364 | 0.8987 |
| base_tilt_rms | 0.0291165 | 0.020242 | 0.6952 |
| base_ang_vel_xy_rms | 0.169111 | 0.126662 | 0.7490 |
| joint_target_step_p95_deg | 3.86431 | 3.24703 | 0.8403 |
| joint_acc_p95_rad_s2 | 29.4307 | 27.1806 | 0.9235 |
| joint_torque_p95_nm | 5.57803 | 5.06599 | 0.9082 |
| mechanical_power_mean_w | 2.98982 | 2.99252 | 1.0009 |
| foot_force_p95_n | 57.4258 | 58.0686 | 1.0112 |
| touchdown_force_p95_n | 78.4555 | 56.9011 | 0.7253 |
| contact_mismatch_mean | 0.0863095 | 0.0952381 | 1.1034 |
| swing_height_mean_m | 0.0375704 | 0.0349546 | 0.9304 |
| action_second_step_p95 | 0.40446 | 0.325598 | 0.8050 |
