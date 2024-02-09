#!/usr/bin/env python3

import numpy as np
from dual_quaternions import DualQuaternion
from pyquaternion import Quaternion
# from .math_tools import dq_log, dx_dt, quat_rot, twist_from_dq_list, vel_from_twist


class ProjectileLaunching(object):
    def __init__(self):
        self.ag = Quaternion(vector=[0, 0, -9.80665])

    @staticmethod
    def correct_poses(dq_i, dq_g, off_0, off_1):
        # dq_d = dq_g * dq_i.quaternion_conjugate()
        # dq_i = off_0 * dq_i * dq_d * off_1 * dq_d.quaternion_conjugate()
        dq_i = off_0 * dq_g * off_1 * dq_g.quaternion_conjugate() * dq_i
        dq_g = off_0 * dq_g * off_1
        return dq_i, dq_g

    @staticmethod
    def estimate_plane_rotation(ng, pg, pt):
        fac = 2 * ng.x * ng.y * pg.x * pg.y +\
              (ng.x**2) * (pg.x**2 - (pt.x**2 + pt.y**2)) +\
              (ng.y**2) * (pg.y**2 - (pt.x**2 + pt.y**2))
        num = (ng.x*pg.x + ng.y*pg.y - np.emath.sqrt(fac)) * 1j
        den = (ng.x + ng.y*1j) * (pt.y + pt.x*1j)
        theta = -np.log(num/den) * 1j
        q_r = Quaternion(axis=[0, 0, 1], angle=np.real(theta))
        # print([fac, num, den, num/den, theta])
        return q_r

    def __optimal_t_impact(self, d_p):
        agz = self.ag.z
        arg = ((4 * d_p.x**2 + 4 * d_p.y**2 + 4 * d_p.z**2) / agz**2)
        res = np.power(arg, 1/4)
        return res

    def optimal_v_launch(self, p_0, p_t):
        d_p = p_t - p_0
        t_f = self.__optimal_t_impact(d_p)
        v_res = (1/t_f)*d_p - .5*t_f*self.ag
        # p_ = self.simulate_launch(t_f, v_res, p_0)
        # print([t_f, v_res])
        return t_f, v_res

    def simulate_launch(self, t_f, v_res, p_0):
        t_ = np.linspace(0, t_f, num=25)
        p_ = [p_0 + (t_i * v_res) + ((.5 * t_i ** 2) * self.ag) for t_i in t_]
        return p_


def gen_movement(r=1.00, n=100):
    # Time vector
    t_vec = np.linspace(0, .8, num=n)

    # Cartesian path
    x = r * (1 - np.cos(.5 * np.pi * t_vec))
    y = r * 0 * t_vec
    z = r * np.sin(.5 * np.pi * t_vec)
    p_vec = np.c_[x, y, z]

    # Quaternion orientation
    q_vec = [(Quaternion(axis=[0, 0, 1], angle=np.pi) *
              Quaternion(axis=[0, 1, 0], angle=-a_i)).elements for a_i in .5 * np.pi * t_vec]

    # Value vectors reshape
    p_vec = np.array(p_vec).reshape((-1, 3))
    q_vec = np.array(q_vec).reshape((-1, 4))

    dq_list = [DualQuaternion.from_quat_pose_array(np.append(q_i, p_i)) for p_i, q_i in zip(p_vec, q_vec)]
    dq_vec = np.array(dq_list, dtype=DualQuaternion)

    t_vec = t_vec / t_vec[-1]

    # Plot values for good measure
    # from matplotlib import pyplot as plt
    # plt.subplot(131)
    # plt.plot(t_vec, p_vec)
    # plt.subplot(132)
    # plt.plot(t_vec, q_vec)
    # plt.subplot(133)
    # plt.plot(t_vec, r_vec)
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(x, y, z)
    # r_vec = [Quaternion(axis=[0, 1, 0], angle=a_i).rotate([0, 0, 1]) for a_i in .5 * np.pi * (t_vec - .0)]
    # r_vec = np.array(r_vec).reshape((-1, 3))
    # ax.quiver(x[::5], y[::5], z[::5], r_vec[::5, 0], r_vec[::5, 1], r_vec[::5, 2], length=.05, normalize=True)
    # ax.axes.set_xlim3d(left=-0.5, right=0.5)
    # ax.axes.set_ylim3d(bottom=-0.5, top=0.5)
    # ax.axes.set_zlim3d(bottom=0., top=1.0)
    # ax.set_proj_type('ortho')
    # plt.show()

    return t_vec, dq_vec


def main():
    pass


if __name__ == '__main__':
    main()
