#!/usr/bin/env python

import numpy as np
from copy import deepcopy as dcp
from dual_quaternions import DualQuaternion
from pyquaternion import Quaternion as Quat
from .operations import quat_rot


order_dict = {'inertial': lambda a, b: a * b,
              'body': lambda a, b: b * a}
forward_mult = order_dict['body']


def dq_exp(dq):
    """
    Dual Quaternion Exponential (se(3) -> SE(3))

    :param DualQuaternion dq: Dual Quaternion
    :return: Dual Quaternion Exponential
    :rtype: DualQuaternion
    """
    exp = dcp(dq)
    exp.q_r = Quat.exp(1 * dq.q_r)
    exp.q_d = (1 * dq.q_d) * Quat.exp(1 * dq.q_r)
    return exp
    

def dq_log(dq):
    """
    Dual Quaternion Logarithm (SE(3) -> se(3))

    :param DualQuaternion dq: Dual Quaternion
    :return: Dual Quaternion Logarithm
    :rtype: DualQuaternion
    """
    log = dcp(dq)
    log.q_r = 1 * Quat.log(dq.q_r)
    log.q_d = 1 * dq.q_d * dq.q_r.inverse
    return log

def next_dq_from_twist(dt, dq, tw, mult=forward_mult):
    """
    Next Dual Quaternion Pose from Current Pose, Current Twist, and Timestep

    :param float dt: Timestep
    :param DualQuaternion dq: Current Pose
    :param DualQuaternion tw: Current Twist
    :return: Next Pose
    :rtype: DualQuaternion
    """
    dq_n = mult(dq_exp(0.5 * dt * tw), dq)
    return dq_n

def twist_from_dq_diff(dt, p_dq, c_dq, mult=forward_mult):
    """
    Current Twist from Previous Pose, Current Pose, and Timestep

    :param float dt: Timestep
    :param DualQuaternion p_dq: Previous Pose
    :param DualQuaternion dq: Current Pose
    :return: Current Twist
    :rtype: DualQuaternion
    """
    tw = (2 / dt) * dq_log(mult(c_dq, p_dq.quaternion_conjugate()))
    return tw

def twist_from_dq_list(t, dq, mult=forward_mult):
    """
    Twist List from Dual Quaternion list

    :param numpy.ndarray t: Time vector
    :param numpy.ndarray dq: Dual Quaternion list
    :return: Twist List
    :rtype: numpy.ndarray
    """
    dt = np.diff(t)
    tw = np.array([twist_from_dq_diff(dti, pdq, cdq, mult=mult) 
                   for cdq, pdq, dti in zip(dq[1:], dq[0:-1], dt)],
                   dtype=DualQuaternion)
    tw = np.append(tw, [tw[-1]])
    return tw

def vel_from_twist(dq, tw):
    """
    Velocity Pure Quaternion from Twist and Pose

    :param DualQuaternion dq: Current Pose
    :param DualQuaternion tw: Current Twist
    :return: Pure Quaternion Velocity
    :rtype: Quaternion
    """
    w = tw.q_r
    p = 2 * dq_log(dq).q_d
    cross = lambda a, b: 0.5 * (a*b - b*a)
    v = tw.q_d - cross(p, w)
    return v


def main():
    pass


if __name__ == '__main__':
    main()
