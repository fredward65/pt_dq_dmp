#!/usr/bin/env python3

import numpy as np
from pyquaternion import Quaternion
from .math_tools import quat_rot


class ProjectileModel(object):
    def __init__(self):
        self.a_g = -9.80665
        self.t_f = 0
        self.dp_0 = Quaternion(vector=np.zeros(3))
        self.n_f = Quaternion(vector=np.zeros(3))
        # Condition Estimator Model
        self.dz0_eq = lambda t_f, z_0, z_f: -(((self.a_g * t_f**2) / 2) - z_f + z_0)/t_f
        self.dx0_eq = lambda t_f, dz_0: (self.n_f.x / self.n_f.z) * (dz_0 + self.a_g * t_f)
        self.dy0_eq = lambda t_f, dz_0: (self.n_f.y / self.n_f.z) * (dz_0 + self.a_g * t_f)
        # Trivial Kinetic Model
        self.x_eq = lambda t, x_0, dx_0: dx_0 * t + x_0
        self.z_eq = lambda t, z_0, dz_0: .5 * self.a_g * t**2 + dz_0 * t + z_0

    def t_f_compute(self, delta_p:Quaternion, n_f:Quaternion):
        d_p_xy = (delta_p.x + delta_p.y)
        d_n_xy = (n_f.x + n_f.y)
        np.seterr(invalid='raise', divide='raise')
        try:
            fac = 2 / (self.a_g * d_n_xy)
            arg = (n_f.z * d_p_xy) - (delta_p.z * d_n_xy)
            res = np.sqrt(fac * arg)
        except FloatingPointError:
            res = np.nan
            print('Not possible')
        return res

    @staticmethod
    def align_n(q_n, dp):
        vec = Quaternion(vector=[0, 0, -1])
        nc = q_n.rotate(vec)
        nc_xy = Quaternion(vector=[nc.x, nc.y, 0])
        dp_xy = Quaternion(vector=[dp.x, dp.y, 0])
        q_rot = quat_rot(nc_xy, dp_xy) * q_n
        n_f = q_rot.rotate(vec)
        return n_f

    def solve(self, p_0, p_f, q_f):
        self.n_f = self.align_n(q_f, p_f - p_0)
        self.t_f = self.t_f_compute(p_f - p_0, self.n_f)
        np.seterr(invalid='raise')
        try:
            dp_0_z = self.dz0_eq(self.t_f, p_0.z, p_f.z)
            dp_0_x = self.dx0_eq(self.t_f, dp_0_z)
            dp_0_y = self.dy0_eq(self.t_f, dp_0_z)
            self.dp_0 = Quaternion(vector=[dp_0_x, dp_0_y, dp_0_z])
        except FloatingPointError:
            self.dp_0 = Quaternion(vector=[np.nan, np.nan, np.nan])
        return self.t_f, self.dp_0

    def evaluate(self, p_0, n=100, t_f=None, dp_0=None):
        t_f = t_f if not t_f is None else self.t_f
        dp_0 = dp_0 if not dp_0 is None else self.dp_0
        t = np.linspace(0, t_f, num=n)
        x = self.x_eq(t, p_0.x, dp_0.x)
        y = self.x_eq(t, p_0.y, dp_0.y)
        z = self.z_eq(t, p_0.z, dp_0.z)
        p = np.array([Quaternion(vector=[x_i, y_i, z_i]) for x_i, y_i, z_i in zip(x, y, z)])
        return p, t


def main():
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    from mpl_toolkits.mplot3d import Axes3D

    np.set_printoptions(precision=3, suppress=True)
    plt.rcParams.update({'text.usetex': True, 'font.size': 7, 'figure.dpi': 100})

    pm = ProjectileModel()

    p_0 = Quaternion(vector=[0, 0, 0])
    p_f = Quaternion(vector=[1, 0, 1])

    fig = plt.figure(1, figsize=(3, 4))
    ax = fig.add_subplot(projection='3d', )
    ax.view_init(elev=0, azim=-90)  # elev=15, azim=-105
    ax.set_xlim3d( 0.0, 1.25)
    ax.set_ylim3d(-.75, .75)
    ax.set_zlim3d( 0.0, 1.25)
    ax.set_box_aspect((1, 1, 1), zoom=1.2)
    ax.set_yticks([])
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_zlabel("$z$", fontsize=12)
    ax.xaxis.labelpad = -5
    ax.yaxis.labelpad = -5
    ax.zaxis.labelpad = -5
    ax.set_proj_type('ortho')
    ax.tick_params(axis='both', which='major', labelsize=7, pad=-2, grid_alpha=0.5)
    ax.xaxis._axinfo["grid"]['linewidth'] = .5
    ax.yaxis._axinfo["grid"]['linewidth'] = .5
    ax.zaxis._axinfo["grid"]['linewidth'] = .5

    for angle in np.linspace(-0.25*np.pi, -0.66*np.pi, num=10):
        print("angle : ", angle)
        q_f = Quaternion(axis=[0, 1, 0], angle=angle)
        n_c = q_f.rotate(Quaternion(vector=[0, 0 , -1]))

        t_f, dp_0 = pm.solve(p_0, p_f, q_f)
        print(dp_0.elements[1:], t_f)

        if not np.isnan(t_f): 
            col = hsv_to_rgb((0.25 + np.abs(angle)/np.pi, 1, 1))
            p, t = pm.evaluate(p_0, n=100)
            txt = r"$\|\dot{\mathbf{p}}_0\|$ : %5.3f $m \cdot s^{-1}$" % dp_0.norm
            ax.plot([p_i.x for p_i in p], [p_i.y for p_i in p], [p_i.z for p_i in p], label=txt, color=col, linewidth=.75)
            ax.quiver(p_0.x, p_0.y, p_0.z, dp_0.x, dp_0.y, dp_0.z, length=.1, color=col, linewidth=.75)
            ax.quiver(p_f.x, p_f.y, p_f.z, n_c.x, n_c.y, n_c.z, length=.2, color=col, linewidth=.75)
    ax.plot(p_0.x, p_0.y, p_0.z, 'ok', markersize=3, zorder=np.inf, label=r"$\mathbf{p}_0$")
    ax.plot(p_f.x, p_f.y, p_f.z, 'xk', markersize=5, zorder=np.inf, label=r"$\mathbf{p}_\mathrm{t}$")
    
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, .01), fontsize=7)
    fig.tight_layout()
    fig.subplots_adjust(left=0, bottom=0)

    # fig.savefig("./src/figures/figure_pmdem.png", dpi=200, bbox_inches="tight")

    plt.subplots_adjust(left=0, bottom=.25, right=1, top=1)
    plt.show()


if __name__ == '__main__':
    main()
