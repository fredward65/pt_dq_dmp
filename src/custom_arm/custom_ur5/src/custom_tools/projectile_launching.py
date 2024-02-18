#!/usr/bin/env python3

import numpy as np
from dual_quaternions import DualQuaternion
from pyquaternion import Quaternion


class ProjectileLaunching(object):
    def __init__(self):
        self.ag = Quaternion(vector=[0, 0, -9.80665])

    @staticmethod
    def correct_poses(dq_i, dq_g, off_0, off_1):
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
        return t_f, v_res

    def simulate_launch(self, t_f, v_res, p_0, n=100):
        t_ = np.linspace(0, t_f, num=n)
        p_ = [p_0 + (t_i * v_res) + ((.5 * t_i ** 2) * self.ag) for t_i in t_]
        return p_, t_


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

    return t_vec, dq_vec


def main():
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    from mpl_toolkits.mplot3d import Axes3D

    np.set_printoptions(precision=3, suppress=True)
    plt.rcParams.update({"text.usetex": True})

    pl = ProjectileLaunching()

    p_0 = Quaternion(vector=[0, 0, 0])

    ax = plt.figure(1).add_subplot(projection='3d')
    ax.view_init(elev=15, azim=-105)
    ax.set_xlim3d( 0.0, 1.0)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d( 0.0, 1.0)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_proj_type('ortho')

    count = 8
    for i in range(count):
        vec_0 = np.multiply(np.random.rand(3) - np.array([0.0, 0.5, 0.0]), [0, 1, 0])
        p_0 = Quaternion(vector=vec_0)
        vec_f = np.multiply(np.random.rand(3) - np.array([0.0, 0.5, 0.0]), [1, 1, 1])
        p_f = Quaternion(vector=vec_f)
        print(p_f.elements[1:])
        
        t_f, dp_0 = pl.optimal_v_launch(p_0, p_f)
        print(dp_0.elements[1:], t_f)

        col = hsv_to_rgb((i/count, .75, .75))
        p, t = pl.simulate_launch(t_f, dp_0, p_0, n=100)
        txt = r"$\|\dot{\mathrm{p}}_0\|$ : %5.3f $m/s$" % dp_0.norm
        ax.plot([p_i.x for p_i in p], [p_i.y for p_i in p], [0 for p_i in p], color=(0,0,0), alpha=0.25, zorder=0)
        ax.plot([p_i.x for p_i in p], [p_i.y for p_i in p], [p_i.z for p_i in p], label=txt, color=col)
        ax.plot(p_f.x, p_f.y, p_f.z, 'x', color=col)
        ax.quiver(p_0.x, p_0.y, p_0.z, dp_0.x, dp_0.y, dp_0.z, length=.05, color=col)

    plt.legend(loc="upper center", ncol=count//4)
    plt.tight_layout()
    plt.show()


def plane_rotation():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    np.set_printoptions(precision=3, suppress=True)
    plt.rcParams.update({"text.usetex": True})

    pl = ProjectileLaunching()

    p_0 = Quaternion(vector=[.5, .5, 1])
    n_0 = Quaternion(vector=[0, -1, 0])

    p_t = Quaternion(vector=[1, 1.5,.5])

    q_r = pl.estimate_plane_rotation(n_0, p_0, p_t)
    n_f = q_r.rotate(n_0)
    p_f = q_r.rotate(p_0)

    ax = plt.figure(1).add_subplot(projection='3d')
    ax.view_init(elev=30, azim=30)
    # ax.view_init(elev=90, azim=-90)
    ax.set_xlim3d( 0.0, 2.0)
    ax.set_ylim3d( 0.0, 2.0)
    ax.set_zlim3d( 0.0, 2.0)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    # ax.set_zticks([])
    ax.set_proj_type('ortho')

    def draw_plane(n, p, color=[0,0,0], label=""):
        p_1 = Quaternion(vector=[0, 0, 1])
        q_1 = (0.5 * (p_1*n - n*p_1)).normalised
        ln1 = np.linspace(-1,   1, num=10)
        ln2 = np.linspace(-.1, 1.5, num=10)
        ln1, ln2 = np.meshgrid(ln1, ln2)
        xx = p.x + p_1.x*ln1 + q_1.x*ln2
        yy = p.y + p_1.y*ln1 + q_1.y*ln2
        zz = p.z + p_1.z*ln1 + q_1.z*ln2
        ax.plot(xx[:, 0], yy[:, 0], zz[:, 0], color=color, alpha=0.33)
        surf = ax.plot_surface(xx, yy, zz, color=color, alpha=0.33, linewidth=10, antialiased=False, label=label)
        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d


    ax.plot(p_0.x, p_0.y, p_0.z, 'xk', label=r"$\mathrm{p}_g$")
    ax.plot(p_f.x, p_f.y, p_f.z, 'xb', label=r"$\mathrm{p}_g'$")
    ax.plot(p_t.x, p_t.y, p_t.z, '*b', label=r"$\mathrm{p}_t$")

    radius = Quaternion(vector=[p_0.x, p_0.y, 0]).norm
    arg_angle = np.linspace(0, 0.5*np.pi, num=100)
    x = radius * np.cos(arg_angle)
    y = radius * np.sin(arg_angle)
    z = p_0.z * np.ones(100)
    ax.plot(x, y, z, 'b', alpha=0.5)

    ax.quiver(p_0.x, p_0.y, p_0.z, n_0.x, n_0.y, n_0.z, length=0.5, colors=[0,0,0], label=r"$\mathrm{n}_g$")
    ax.quiver(p_f.x, p_f.y, p_f.z, n_f.x, n_f.y, n_f.z, length=0.5, colors=[0,0,1], label=r"$\mathrm{n}_g\prime$")
    draw_plane(n_0, p_0, color=[1,0,0], label=r"$\underline{\pi}$")
    draw_plane(n_f, p_f, color=[0,0,1], label=r"$\underline{\pi}_g\prime$")

    plt.legend(loc="center", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # main()
    plane_rotation()
