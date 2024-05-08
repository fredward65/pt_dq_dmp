#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from custom_tools.projectile_model import ProjectileModel
from custom_tools.projectile_throwing import ProjectileThrowing
from custom_tools.math_tools import quat_rot, dx_dt
from matplotlib import animation
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion
from rospkg import RosPack

np.set_printoptions(precision=3, suppress=True)
plt.rcParams.update({'text.usetex': True, 'font.size': 7, 'figure.dpi': 300})
plt.style.use('dark_background')

file_path = RosPack().get_path("custom_ur5") + "/resources/"


def arbitrary():
    pm = ProjectileModel()

    p_0 = Quaternion(vector=[0, 0, 0])
    p_f = Quaternion(vector=[1, 0, 1])

    fig = plt.figure(1, figsize=(3, 3))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=15, azim=-60)  # elev=15, azim=-105
    ax.set_xlim3d( 0.0, 1.25)
    ax.set_ylim3d(-.75, .75)
    ax.set_zlim3d( 0.0, 1.25)
    ax.set_box_aspect((1, 1, 1), zoom=1.2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_zlabel("$z$", fontsize=12)
    ax.xaxis.labelpad = -15
    ax.yaxis.labelpad = -15
    ax.zaxis.labelpad = -15
    ax.set_proj_type('ortho')
    ax.tick_params(axis='both', which='major', labelsize=7, pad=-2, grid_alpha=0.5)
    ax.xaxis.set_pane_color((1,1,1,0))
    ax.yaxis.set_pane_color((1,1,1,0))
    ax.zaxis.set_pane_color((1,1,1,0))
    ax.xaxis._axinfo["grid"]['linewidth'] = .1
    ax.yaxis._axinfo["grid"]['linewidth'] = .1
    ax.zaxis._axinfo["grid"]['linewidth'] = .1

    lines = []
    trajs = []
    for i, angle in enumerate(np.linspace(-0.2*np.pi, -0.6*np.pi, num=10)):
        y_pos = 1 - 2*i/10
        z_pos = p_f.z # (.5*(.2+(angle/np.pi))/.4) + 1
        p_0 = Quaternion(vector=[p_0.x, y_pos, p_0.z])
        p_f = Quaternion(vector=[p_f.x, y_pos, z_pos])
        print("angle : ", angle)
        q_f = Quaternion(axis=[0, 1, 0], angle=angle)
        n_c = q_f.rotate(Quaternion(vector=[0, 0 , -1]))

        t_f, dp_0 = pm.solve(p_0, p_f, q_f)
        print(dp_0.elements[1:], t_f)

        if not np.isnan(t_f): 
            col = hsv_to_rgb((0.25 + np.abs(angle)/np.pi, .75, 1))
            p, t = pm.evaluate(p_0, n=int(np.floor(200*t_f)+1))
            txt = r"$\|\dot{\mathbf{p}}_0\|$ : %5.3f $m \cdot s^{-1}$" % dp_0.norm
            
            ax.plot(p_0.x, p_0.y, p_0.z, 'ow', markersize=2, zorder=np.inf, color=col)
            ax.plot(p_f.x, p_f.y, p_f.z, 'ow', markersize=2, zorder=np.inf)
            ax.plot(p_f.x, p_f.y, 0, 'ow', markersize=2, zorder=np.inf, alpha=.5)
            ax.quiver(p_0.x, p_0.y, 0, dp_0.x, dp_0.y, 0, length=.05, color=col, linewidth=1, alpha=.5)
            line1 = ax.plot(p[0].x, p[0].y, 0, label=txt, color='w', linewidth=1, alpha=.5)[0]
            line2 = ax.plot(p[0].x, p[0].y, p[0].z, label=txt, color=col, linewidth=.75)[0]
            quiver1 = ax.quiver(p_0.x, p_0.y, p_0.z, dp_0.x, dp_0.y, dp_0.z, length=.05, color=col, linewidth=1)
            quiver2 = ax.quiver(p_f.x, p_f.y, p_f.z, n_c.x, n_c.y, n_c.z, length=.2, color=col, linewidth=.75)
            quiver2.set_color((1,1,1,.5))
            
            trajs.append(p)
            lines.append([line1, line2, quiver1, quiver2, col])
        else:
            print('ERROR')
    

    # ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, .01), fontsize=7)
    fig.tight_layout()
    fig.subplots_adjust(left=0, bottom=0)
    frame_len = np.max([len(traj) for traj in trajs])

    def update(frame):
        idx = frame + 1 if frame < frame_len else frame_len
        ax.view_init(elev=15, azim=-60 - 10*idx/frame_len)
        for traj, line in zip(trajs, lines):
            if frame < traj.shape[0]:
                x_data = [p_i.x for p_i in traj[:idx]]
                y_data = [p_i.y for p_i in traj[:idx]]
                z_data = [p_i.z for p_i in traj[:idx]]
                z_zero = [0 for i in range(idx)]
                line[0].set_data_3d((x_data, y_data, z_zero))
                line[1].set_data_3d((x_data, y_data, z_data))
            else:
                line[-2].set_color(line[-1])
                line[-2].set_alpha(1)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=frame_len+5, interval=20, repeat=False)
    filename = file_path + "pm_animation"
    ani.save(filename="%s.avi" % filename, writer="ffmpeg")

    # plt.subplots_adjust(left=0, bottom=.25, right=1, top=1)
    plt.show()


def optimal():
    pl = ProjectileThrowing()

    p_0 = Quaternion(vector=[0, 0, 0])
    p_f = Quaternion(vector=[1, 0, 1])

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=25, azim=-135)
    ax.set_xlim3d( 0.0, 1.0)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d( 0.0, 1.0)
    ax.set_box_aspect((1, 1, 1), zoom=1.2)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_zlabel("$z$", fontsize=12)
    ax.xaxis.set_pane_color((1,1,1,0))
    ax.yaxis.set_pane_color((1,1,1,0))
    ax.zaxis.set_pane_color((1,1,1,0))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.labelpad = -15
    ax.yaxis.labelpad = -15
    ax.zaxis.labelpad = -15
    ax.xaxis._axinfo["grid"]['linewidth'] = .1
    ax.yaxis._axinfo["grid"]['linewidth'] = .1
    ax.zaxis._axinfo["grid"]['linewidth'] = .1
    ax.tick_params(axis='both', which='major', pad=-2)
    ax.set_proj_type('ortho')

    trajs = []
    lines = []
    for i, angle in enumerate(np.linspace(-0.2*np.pi, -0.6*np.pi, num=10)):
        y_pos = (.5 + (.2+(angle/np.pi))/.4)
        z_pos = (.5*(.2+(angle/np.pi))/.4) + 1
        p_0 = Quaternion(vector=[p_0.x, y_pos, p_0.z])
        p_f = Quaternion(vector=[p_f.x, y_pos, z_pos])
        
        t_f, dp_0 = pl.optimal_v_launch(p_0, p_f)
        print(dp_0.elements[1:], t_f)

        col = hsv_to_rgb((i/10, .75, .75))
        p, t = pl.simulate_launch(t_f, dp_0, p_0, n=int(np.floor(200*t_f)+1))
        txt = r"$\|\dot{\mathbf{p}}_0\|$ : %5.3f $m \cdot s^{-1}$" % dp_0.norm
        line1 = ax.plot([p_i.x for p_i in p], [p_i.y for p_i in p], [0 for p_i in p], color='w', alpha=0.5, zorder=0, linewidth=.75)[0]
        line2 = ax.plot([p_i.x for p_i in p], [p_i.y for p_i in p], [p_i.z for p_i in p], label=txt, color=col, linewidth=1)[0]
        ax.quiver(p_0.x, p_0.y, p_0.z, dp_0.x, dp_0.y, 0, length=.05, color=col, linewidth=1, alpha=.5)
        ax.quiver(p_0.x, p_0.y, p_0.z, dp_0.x, dp_0.y, dp_0.z, length=.05, color=col, linewidth=1)
        ax.plot(p_0.x, p_0.y, p_0.z, 'o', markersize=3, color=col)
        ax.plot(p_f.x, p_f.y, 0, 'o', markersize=3, color='w', alpha=.5)
        ax.plot(p_f.x, p_f.y, p_f.z, 'o', markersize=3, color='w')

        trajs.append(p)
        lines.append([line1, line2])
    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, .0), ncol=count//4)
    fig.tight_layout()
    fig.subplots_adjust(left=0, bottom=0)

    frame_len = np.max([len(traj) for traj in trajs])
    def update(frame):
        idx = frame + 1 if frame < frame_len else frame_len
        ax.view_init(elev=15, azim=-60 - 10*idx/frame_len)
        for traj, line in zip(trajs, lines):
            if frame < len(traj):
                x_data = [p_i.x for p_i in traj[:idx]]
                y_data = [p_i.y for p_i in traj[:idx]]
                z_data = [p_i.z for p_i in traj[:idx]]
                z_zero = [0 for i in range(idx)]
                line[0].set_data_3d((x_data, y_data, z_zero))
                line[1].set_data_3d((x_data, y_data, z_data))

    ani = animation.FuncAnimation(fig=fig, func=update, frames=frame_len, interval=20, repeat=False)
    filename = file_path + "pm_opt_animation"
    ani.save(filename="%s.avi" % filename, writer="ffmpeg")

    # plt.subplots_adjust(left=0, bottom=.25, right=1, top=1)
    plt.show()


def plane_rot():
    from matplotlib.ticker import MultipleLocator
    from mpl_toolkits.mplot3d import Axes3D

    pl = ProjectileThrowing()

    p_0 = Quaternion(vector=[.5, .5, 1])
    n_0 = Quaternion(vector=[0, -1, 0])

    p_t = Quaternion(vector=[1, 1.5,.5])

    q_r = pl.estimate_plane_rotation(n_0, p_0, p_t)

    fig = plt.figure(1, figsize=(3, 3))
    ax = fig.add_subplot(projection='3d')
    ax.tick_params(axis='both', which='major', labelsize=7, pad=-2, grid_alpha=0.5)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.zaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis._axinfo["grid"]['linewidth'] = .1
    ax.yaxis._axinfo["grid"]['linewidth'] = .1
    ax.zaxis._axinfo["grid"]['linewidth'] = .1
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.labelpad = -15
    ax.yaxis.labelpad = -15
    ax.zaxis.labelpad = -15
    ax.xaxis.set_pane_color((1,1,1,0))
    ax.yaxis.set_pane_color((1,1,1,0))
    ax.zaxis.set_pane_color((1,1,1,0))

    ax.set_proj_type('ortho')
    ax.set_xlim3d( 0.0, 2.0)
    ax.set_ylim3d( 0.0, 2.0)
    ax.set_zlim3d( 0.0, 2.0)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_zlabel("$z$", fontsize=12)

    ax.view_init(elev=30, azim=30)
    ax.set_box_aspect((1, 1, 1))

    def get_plane_data(n, p):
        p_1 = Quaternion(vector=[0, 0, 1])
        q_1 = (0.5 * (p_1*n - n*p_1)).normalised
        ln1 = np.linspace(-1,   1, num=10)
        ln2 = np.linspace(-.1, 1.5, num=10)
        ln1, ln2 = np.meshgrid(ln1, ln2)
        xx = p.x + p_1.x*ln1 + q_1.x*ln2
        yy = p.y + p_1.y*ln1 + q_1.y*ln2
        zz = p.z + p_1.z*ln1 + q_1.z*ln2
        return xx, yy, zz

    def draw_plane(n, p, color='w', label=""):
        xx, yy, zz = get_plane_data(n, p)
        line = ax.plot(xx[:, 0], yy[:, 0], zz[:, 0], color=color, alpha=0.5)[0]
        surf = ax.plot_surface(xx, yy, zz, color=color, alpha=0.5, linewidth=10, antialiased=True, label=label)
        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d
        return line, surf
    
    q_r_i = Quaternion(angle=0, axis=q_r.axis)
    n_f = q_r_i.rotate(n_0)
    p_f = q_r_i.rotate(p_0)

    draw_plane(n_0, p_0, color='m')
    global pline, psurf, pquiver
    pline, psurf = draw_plane(n_f, p_f, color='c')

    ax.plot(p_0.x, p_0.y, 0, 'ow', markersize=2)
    ax.plot(p_0.x, p_0.y, p_0.z, 'om', markersize=2)
    ppoint1 = ax.plot(p_f.x, p_f.y, 0, 'ow', markersize=2)[0]
    ppoint2 = ax.plot(p_f.x, p_f.y, p_f.z, 'oc', markersize=2)[0]

    ax.quiver(p_0.x, p_0.y, p_0.z, n_0.x, n_0.y, n_0.z, length=0.5, colors='m', linewidth=.75)
    pquiver = ax.quiver(p_f.x, p_f.y, p_f.z, n_f.x, n_f.y, n_f.z, length=0.5, colors='c', linewidth=.75)

    ax.plot(p_t.x, p_t.y,     0, 'ow', markersize=3, alpha=0.5)
    ax.plot(p_t.x, p_t.y, p_t.z, 'oc', markersize=3)

    # radius = Quaternion(vector=[p_0.x, p_0.y, 0]).norm
    # arg_angle = np.linspace(0, 0.5*np.pi, num=100)
    # x = radius * np.cos(arg_angle)
    # y = radius * np.sin(arg_angle)
    # z = p_0.z * np.ones(100)
    # ax.plot(x, y, z, 'c', alpha=0.5)

    # fig.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.subplots_adjust(left=0, bottom=0)

    frame_len = 100 # np.max([len(traj) for traj in trajs])

    def update(frame):
        global pline, psurf, pquiver

        idx = frame + 1
        ax.view_init(elev=30 - 10*idx/frame_len, azim=30 - 20*idx/frame_len)

        ax.set_box_aspect((1, 1, 1), zoom=1 + .1*idx/frame_len)
        
        if frame < frame_len:
            q_r_i = Quaternion(angle=q_r.angle*idx/frame_len, axis=q_r.axis)
            n_f = q_r_i.rotate(n_0)
            p_f = q_r_i.rotate(p_0)

            ppoint1.set_data_3d(([p_f.x], [p_f.y], [0]))
            ppoint2.set_data_3d(([p_f.x], [p_f.y], [p_f.z]))

            # xx, yy, zz = get_plane_data(n_f, p_f)
            # pline.set_data_3d((xx[:, 0], yy[:, 0], zz[:, 0]))
            pline.remove()
            psurf.remove()
            pquiver.remove()
            pline, psurf = draw_plane(n_f, p_f, color='c')
            pquiver = ax.quiver(p_f.x, p_f.y, p_f.z, n_f.x, n_f.y, n_f.z, length=0.5, colors='c', linewidth=.75)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=frame_len*2, interval=20, repeat=False)
    filename = file_path + "pln_rot_animation"
    # ani.save(filename="%s.avi" % filename, writer="ffmpeg")

    plt.show()


def main():
    # arbitrary()
    # optimal()
    plane_rot()


if __name__ == '__main__':
    main()
