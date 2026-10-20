"""ROS 2 /cmd_vel + held Joy button -> local policy command transport."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys

from bennett_deploy.ros_command import DEFAULT_PORT, JoystickVelocity, TeleopGate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-only", action="store_true", help="Use already running Joy/cmd_vel publishers")
    parser.add_argument("--device-name", default="TGZ Controller")
    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--joy-topic", default="/joy")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=0.25)
    parser.add_argument("--deadman-button", type=int, default=4)
    parser.add_argument("--stop-button", type=int, default=2)
    parser.add_argument("--axes", type=int, nargs=3, default=(3, 4, 0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--scales", type=float, nargs=3, default=(0.25, 0.12, 0.5), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--deadzone", type=float, default=0.05)
    args, ros_args = parser.parse_known_args(argv)
    gate = TeleopGate(args.timeout, args.deadman_button, args.stop_button)
    mapper = JoystickVelocity(args.axes, args.scales, args.deadzone)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import Joy
        from rclpy.qos import QoSProfile, ReliabilityPolicy
    except ImportError as exc:
        parser.exit(2, f"ROS 2 Python environment required: {exc}\nUse start_ros2_teleop.sh on Jetson.\n")

    children = []
    transport = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    node = None
    try:
        rclpy.init(args=ros_args)
        node = rclpy.create_node("bennett_cmd_bridge")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(Twist, args.topic, lambda m: gate.update_command((m.linear.x, m.linear.y, m.angular.z)), qos)
        publisher = None if args.bridge_only else node.create_publisher(Twist, args.topic, 1)

        def joy_callback(message):
            gate.update_joy(message.buttons)
            if publisher is not None:
                command = mapper.command(message.axes)
                twist = Twist()
                twist.linear.x, twist.linear.y, twist.angular.z = command
                publisher.publish(twist)

        node.create_subscription(Joy, args.joy_topic, joy_callback, qos)

        def send():
            if any(child.poll() is not None for child in children):
                raise RuntimeError("Joystick child process exited; bridge stopped")
            transport.sendto(gate.packet(), ("127.0.0.1", args.port))

        node.create_timer(0.02, send)
        if not args.bridge_only:
            for command in (
                ["ros2", "run", "joy", "joy_node", "--ros-args", "-p", f"device_name:={args.device_name}", "-p", "autorepeat_rate:=50.0", "-p", "deadzone:=0.0"],
            ):
                children.append(subprocess.Popen(command, start_new_session=(os.name != "nt")))
        node.get_logger().info(
            f"{args.topic} -> 127.0.0.1:{args.port}; hold Joy buttons[{args.deadman_button}] for velocity; "
            f"buttons[{args.stop_button}] requests policy shutdown; timeout={args.timeout}s"
        )
        if not args.bridge_only:
            node.get_logger().info(f"Proportional Joy axes vx/vy/wz={args.axes}; full-stick scales={args.scales}; deadzone={args.deadzone}")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        gate.held = False
        transport.sendto(gate.packet(), ("127.0.0.1", args.port))
        transport.close()
        for child in children:
            if child.poll() is None:
                if os.name == "nt":
                    child.terminate()
                else:
                    os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        child.kill()
                    else:
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
