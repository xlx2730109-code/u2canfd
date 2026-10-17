"""Stable entry point for the Bennett go2-10 omnidirectional hardware policy."""

# D:\Conda\envs\env_isaaclab\python.exe -B quad_leg_xu\quad_leg_go2-10_deploy.py --keyboard --imu_port COM4
# python quad_leg_xu\quad_leg_go2-10_deploy.py --keyboard --imu_port COM4

from bennett_deploy.runtime import main


if __name__ == "__main__":
    raise SystemExit(main())
