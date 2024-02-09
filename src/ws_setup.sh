. /opt/ros/${ROS_DISTRO}/setup.bash

rosdep install --from-paths ./src --ignore-src -r -y

catkin_make

. devel/setup.bash