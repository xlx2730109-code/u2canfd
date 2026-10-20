#!/usr/bin/env bash
set -e
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
onboard_dir=$(cd "$script_dir/../.." && pwd)
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-73}
export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-1}
if [ -f "$onboard_dir/ros_ws/install/setup.bash" ]; then
    source "$onboard_dir/ros_ws/install/setup.bash"
fi
exec /usr/bin/python3 "$script_dir/ros2_teleop.py" "$@"
