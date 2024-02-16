#!/usr/bin/env python3

import moveit_commander
import numpy as np
import rospkg
import rospy
import sys
from copy import deepcopy
from dual_quaternions import DualQuaternion
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import DeleteModel, SetModelState, SpawnModel
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import DisplayTrajectory
from pyquaternion import Quaternion


class ArmManager(object):
    def __init__(self, p_offset=Quaternion(np.zeros(4)), q_offset=Quaternion([1, 0, 0, 0]), planner:str='ProjEST'):
        """
        Arm Manager Constructor
        """
        # Initialise moveit_commander and rosnode
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('movement_sandbox', anonymous=False)
        rospy.on_shutdown(self.delete_gazebo_balls)

        # Wait for Gazebo to respond
        rospy.wait_for_message('/gazebo/model_states', ModelStates)

        # Initialise robot and move groups
        self.robot = moveit_commander.robot.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group = moveit_commander.move_group.MoveGroupCommander("ur5_arm")
        self.arm_group.set_planner_id(planner)  # LazyPRMstar  RRTstar PDST ProjEST

        p = PoseStamped()
        p.header.frame_id = self.robot.get_planning_frame()
        p.pose.position.x =  .0
        p.pose.position.y =  .0
        p.pose.position.z = -.5
        self.scene.add_box("table", p, (2.0, 2.0, 1.0))

        # Joint space variables
        self.j_home = self.arm_group.get_current_state().joint_state
        self.j_home.name = list(self.j_home.name)[:6]
        self.j_home.position = [self.arm_group.get_named_target_values('home')[name_i] for name_i in self.j_home.name]

        # Cartesian space variables
        self.p_offset = p_offset
        self.q_offset = q_offset
        dq_off_p = DualQuaternion.from_quat_pose_array(np.append([1, 0, 0, 0], self.p_offset.elements[1:]))
        dq_off_q = DualQuaternion.from_quat_pose_array(np.append(self.q_offset.elements, [0, 0, 0]))
        self.dq_offset = dq_off_q * dq_off_p
        self.current_pose = None
        self.step = 0.05

        self.frac_lim = .25

        # Diagnostic publisher for waypoint poses
        # self.pose_pub = rospy.Publisher('/checker',PoseStamped,latch=True,queue_size=5)
        # Publish trajectory in RViz
        # self.display_traj_pub = rospy.Publisher('/move_group/display_planned_path', DisplayTrajectory, queue_size=20)

    @staticmethod
    def traj_time_scale(plan, scale):
        """
        Scale the time of a MoveIt! Trajectory

        :param moveit_msgs.msg._RobotTrajectory.RobotTrajectory plan: MoveIt! Trajectory Plan
        :param float scale: Scale
        :return: Time-scaled Trajectory Plan
        :rtype: moveit_msgs.msg._RobotTrajectory.RobotTrajectory
        """
        for i, plan_i in enumerate(plan.joint_trajectory.points):
            plan.joint_trajectory.points[i].velocities = tuple(0 * np.array(plan_i.velocities))
            plan.joint_trajectory.points[i].accelerations = tuple(0 * np.array(plan_i.accelerations))
            plan.joint_trajectory.points[i].time_from_start *= scale
        return plan

    @staticmethod
    def execute_group_plan(group, plan):
        """
        Execute plan using MoveIt! group
        """
        # success = None
        # while not success:
        group.set_goal_joint_tolerance(0.05)
        success = group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        return success

    @staticmethod
    def pose_from_dq(dq):
        """
        Geometry Pose from Dual Quaternion Pose

        :param DualQuaternion dq: Dual Quaternion Pose
        :return: Geometry Pose
        :rtype: geometry_msgs.msg.Pose
        """
        pose = Pose()
        # Set Pose Orientation
        pose.orientation.w = dq.q_r.w
        pose.orientation.x = dq.q_r.x
        pose.orientation.y = dq.q_r.y
        pose.orientation.z = dq.q_r.z
        # Set Pose Position
        pos = dq.translation()
        pose.position.x = pos[0]
        pose.position.y = pos[1]
        pose.position.z = pos[2]
        return pose

    def move_pose(self, dq:DualQuaternion, t_scale:float=1.0):
        """
        Move TCP to given pose via Inverse Kinematics (IK)

        :param DualQuaternion dq: Dual Quaternion Pose
        :param float t_scale: Time scale factor
        """
        dq = dq * self.dq_offset
        pose_goal = self.pose_from_dq(dq)
        flag = None
        success = None
        while not success and not flag:
            plan = self.arm_group.go(wait=True)
            self.arm_group.set_pose_target(pose_goal)
            flag, plan, _, _ = self.arm_group.plan()
            plan = self.traj_time_scale(plan, t_scale)
            success = self.arm_group.execute(plan, wait=True)
            self.arm_group.stop()  # To guarantee no residual movement
            self.arm_group.clear_pose_targets()
        return flag

    def move_l(self, dq, t_scale=1.0):
        """
        Move linearly to given pose

        :param dq:
        :param t_scale:
        """
        waypoints = [deepcopy(self.arm_group.get_current_pose().pose), self.pose_from_dq(dq)]
        self.follow_waypoints(waypoints, t_scale=t_scale)

    def delete_duplicates(self, plan):
        # Time vector
        l_ = len(plan.joint_trajectory.points) - 1
        t_f = np.array([t_i.time_from_start.to_sec() for t_i in plan.joint_trajectory.points])
        for i, (p_t, c_t) in enumerate(zip(t_f[1:][::-1], t_f[:-1][::-1])):
            if c_t == p_t:
                dup = plan.joint_trajectory.points.pop(l_ - i).time_from_start.to_sec()
                # print(dup)
        # print(np.max(np.diff([p_i.positions[-1] for p_i in plan.joint_trajectory.points])))
        return plan

    def reassign_time(self, plan, t_vec):
        # Time vector
        t_f = np.array([t_i.time_from_start.to_sec() for t_i in plan.joint_trajectory.points])
        t_l = t_f.shape[0] if t_f.shape[0] < t_vec.shape[0] else t_vec.shape[0]
        n_points = []
        for i in range(t_l):
            n_points.append(plan.joint_trajectory.points[i])
            n_points[i].time_from_start = rospy.Time.from_sec(t_vec[i])
        plan.joint_trajectory.points = n_points
        return plan

    def waypoints_from_dq(self, dq_list):
        waypoints = []
        w_pose = self.arm_group.get_current_pose().pose
        waypoints.append(deepcopy(w_pose))
        for dq_i in dq_list:
            dq_i = dq_i * self.dq_offset
            p_i = dq_i.translation()
            q_i = dq_i.q_r
            w_pose.position.x = p_i[0]
            w_pose.position.y = p_i[1]
            w_pose.position.z = p_i[2]
            w_pose.orientation.w = q_i.w
            w_pose.orientation.x = q_i.x
            w_pose.orientation.y = q_i.y
            w_pose.orientation.z = q_i.z
            waypoints.append(deepcopy(w_pose))
        return waypoints

    def plan_waypoints(self, dq_list, t_vec):
        waypoints = self.waypoints_from_dq(dq_list)
        (plan, fraction) = self.arm_group.compute_cartesian_path(waypoints, self.step, 0.0)
        print('Cartesian path computed at %5.2f%%' % (fraction * 100))
        print('Plan has %i points' % len(plan.joint_trajectory.points))
        if fraction > self.frac_lim:
            # plan = self.traj_time_scale(plan, t_scale)
            # self.move_pose(dq_list[0], t_scale=0.25)
            plan = self.reassign_time(plan, t_vec)
            plan = self.delete_duplicates(plan)
        return plan, fraction

    def follow_waypoints(self, dq_list, t_vec):
        plan, fraction = self.plan_waypoints(dq_list, t_vec)
        success = None
        if fraction > self.frac_lim:
            success = self.execute_group_plan(self.arm_group, plan)
        return success

    def try_launch(self, dq_traj, t_vec, id_=0):
        dq_0 = dq_traj[0]
        fraction = -1
        # Move to initial pose
        if self.move_pose(dq_0, t_scale=.25):
            plan = None
            while fraction < .9:
                self.arm_group.set_goal_joint_tolerance(0.05)
                waypoints = self.waypoints_from_dq(dq_traj)
                (plan, fraction) = self.arm_group.compute_cartesian_path(waypoints, 0.05, 0.0)
                print('Cartesian path computed at %5.2f%%' % (fraction * 100))
                print('Plan has %i points' % len(plan.joint_trajectory.points))
            # Spawn Ball
            dq_ = dq_0 * DualQuaternion.from_quat_pose_array([1, 0, 0, 0, .00, .00, .03])
            self.load_gazebo_ball(dq_, id_=id_)
            rospy.sleep(1)
            # Process trajectory plan
            plan = self.reassign_time(plan, t_vec)
            plan = self.delete_duplicates(plan)
            # Launch
            self.execute_group_plan(self.arm_group, plan)

    def go_home(self, t_scale=1.0):
        # success = None
        # while not success:
        _, plan, _, _ = self.arm_group.plan(self.j_home)
        plan = self.traj_time_scale(plan, t_scale)
        success = self.arm_group.execute(plan, wait=True)
        self.arm_group.stop()

    def load_gazebo_ball(self, dq, id_=0, reference_frame="world"):
        # Get Pose from dq
        pose = self.pose_from_dq(dq)
        # Get Model's Path
        model_path = rospkg.RosPack().get_path('custom_ur5') + "/models/"
        # Load Block URDF
        block_xml = ''
        with open(model_path + "ball.urdf", "r") as block_file:
            block_xml = block_file.read().replace('\n', '')
        # Spawn Block URDF
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        try:
            self.current_object = "ball_%i" % id_
            self.current_pose = pose
            spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
            resp_urdf = spawn_urdf(self.current_object, block_xml, "/", pose, reference_frame)
        except rospy.ServiceException as e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))

    def get_ball_position(self, id_=0):
        return self.get_gazebo_object_pose("ball_%i" % id_).position

    @staticmethod
    def get_gazebo_object_pose(obj_name):
        obj_pose = None
        flag_table = False
        while not flag_table:
            gazebo_msg = rospy.wait_for_message("/gazebo/model_states", ModelStates)
            flag_table = True if obj_name in gazebo_msg.name else False
            if flag_table:
                idx = gazebo_msg.name.index(obj_name)
                obj_pose = gazebo_msg.pose[idx]
        return obj_pose

    @staticmethod
    def delete_gazebo_balls():
        """
        Delete Gazebo Models on ROS Exit
        """
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            gazebo_msg = rospy.wait_for_message("/gazebo/model_states", ModelStates)
            for name in gazebo_msg.name:
                if name.find('ball') >= 0:
                    resp_delete = delete_model(name)
        except rospy.ServiceException as e:
            rospy.loginfo("Delete Model service call failed: {0}".format(e))


def main():
    pass


if __name__ == "__main__":
    main()
