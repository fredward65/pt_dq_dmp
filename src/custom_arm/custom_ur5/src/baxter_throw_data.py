#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from custom_tools.math_tools import npa_to_dql, dq_log, pose_from_dq
# from dual_quaternions import DualQuaternion
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion
from rospkg import RosPack

np.set_printoptions(precision=3, suppress=True)
plt.rcParams.update({'text.usetex': True, 'font.size': 7, 'figure.dpi': 150})


def draw_axes(ax, dq_list, l_=.05):
    for dq_i in dq_list:
        r_i = dq_i.q_r
        p_i = 2 * dq_log(dq_i).q_d
        u_i = r_i.rotate(Quaternion(vector=[1, 0, 0]))
        v_i = r_i.rotate(Quaternion(vector=[0, 1, 0]))
        w_i = r_i.rotate(Quaternion(vector=[0, 0, 1]))
        ax.quiver(p_i.x, p_i.y, p_i.z, u_i.x, u_i.y, u_i.z, length=l_, color=[1,0,0], linewidth=.5, alpha=.5)
        ax.quiver(p_i.x, p_i.y, p_i.z, v_i.x, v_i.y, v_i.z, length=l_, color=[0,1,0], linewidth=.5, alpha=.5)
        ax.quiver(p_i.x, p_i.y, p_i.z, w_i.x, w_i.y, w_i.z, length=l_, color=[0,0,1], linewidth=.5, alpha=.5)


def main():
    file_path = RosPack().get_path("custom_ur5") + "/resources/"
    file_name = "baxter_throw_data.npz"

    data = np.load(file_path + file_name)
    print(data.files)

    p_target = Quaternion(vector=data['target'])
    tau = data['params'][-1]
    t_d = data['td']
    dqg_d = data['goal']
    dq_d = data['dqd']
    t_f = data['tf']
    dq_f = data['dqf']
    t_t = data['tt']
    dq_t = data['dqt']
    dq_r = data['dqtrue']

    print(p_target)
    print(data['params'])

    fig = plt.figure(figsize=(3, 2.5), layout='tight')
    axs = fig.subplots(4, 2)
    axs[0][0].set_title(r'$\mathbf{q}_\mathrm{r}$')
    axs[0][1].set_title(r'$\mathbf{q}_\mathrm{t}$')
    labels = ['$w$', '$x$', '$y$', '$z$']
    for i in range(4):
        axs[i][0].plot(t_f, dq_f[:, i + 0], 'b', linewidth=.8)
        axs[i][1].plot(t_f, dq_f[:, i + 4], 'b', linewidth=.8, label=r'$\underline{\mathbf{q}}_{\mathrm{g}_\mathrm{f\ theory}}$' if i == 3 else None)

        axs[i][0].plot(t_t, dq_t[:, i + 0], '-.', color='tab:pink', linewidth=.8)
        axs[i][1].plot(t_t, dq_t[:, i + 4], '-.', color='tab:pink', linewidth=.8, label=r'$\underline{\mathbf{q}}_{\mathrm{g}_\mathrm{f\ real\ time}}$' if i == 3 else None)
        
        axs[i][0].plot(t_t, dq_r[:, i + 0], 'r', linewidth=.8)
        axs[i][1].plot(t_t, dq_r[:, i + 4], 'r', linewidth=.8, label=r'$\underline{\mathbf{q}}_{\mathrm{g}_\mathrm{f\ robot}}$' if i == 3 else None)
        
        axs[i][0].plot(t_d, dq_d[:, i + 0], '--k', linewidth=.8)
        axs[i][1].plot(t_d, dq_d[:, i + 4], '--k', linewidth=.8, label=r'$\underline{\mathbf{q}}_{\mathrm{g}_\mathrm{d}}$' if i == 3 else None)

        ylims = axs[i][0].get_ylim()
        axs[i][0].vlines(tau * t_d[-1], ylims[0], ylims[-1], color=(0,0,1), linewidth=.5, linestyle='dotted')
        ylims = axs[i][1].get_ylim()
        axs[i][1].vlines(tau * t_d[-1], ylims[0], ylims[-1], color=(0,0,1), linewidth=.5, linestyle='dotted', label=r'$\tau \mathrm{T}$' if i == 3 else None)
        axs[i][0].hlines(dqg_d[i + 0], 0, t_f[-1], color=(1,0,.5), linewidth=.5, linestyle='dotted')
        axs[i][1].hlines(dqg_d[i + 4], 0, t_f[-1], color=(1,0,.5), linewidth=.5, linestyle='dotted', label=r'$\underline{\mathbf{q}}_{\mathrm{g}_\mathrm{f}}$' if i == 3 else None)
        
        axs[i][0].set_ylabel(labels[i], fontsize=10)
        axs[i][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[i][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        if i < 3:
            axs[i][0].set_xticks([])
            axs[i][1].set_xticks([])
        else:
            axs[i][0].set_xlabel('$t$')
            axs[i][1].set_xlabel('$t$')
    fig.legend(ncols=3, loc="lower center", bbox_to_anchor=(.5, .0), fontsize=6)
    fig.tight_layout(rect=[0, .13, 1, 1])

    fig_3d = plt.figure(figsize=(3, 3), tight_layout=True)
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.set_title(r"$\underline{\mathbf{q}}$", fontsize=12)

    dq_d_ = npa_to_dql(dq_d)
    dq_f_ = npa_to_dql(dq_f)
    dq_t_ = npa_to_dql(dq_t)
    dq_r_ = npa_to_dql(dq_r)
    q_d, p_d = pose_from_dq(dq_d_)
    q_f, p_f = pose_from_dq(dq_f_)
    q_t, p_t = pose_from_dq(dq_t_)
    q_r, p_r = pose_from_dq(dq_r_)

    ax_3d.plot(p_d[:, 0], p_d[:, 1], p_d[:, 2], '--k', linewidth=1)
    ax_3d.plot(p_d[:, 0], p_d[:, 1], 0*p_d[:, 2] - .5, 'k', linewidth=1, alpha=0.25)
    ax_3d.plot(p_f[:, 0], p_f[:, 1], p_f[:, 2], 'b', linewidth=1)
    ax_3d.plot(p_f[:, 0], p_f[:, 1], 0*p_f[:, 2] - .5, 'k', linewidth=1, alpha=0.25)
    ax_3d.plot(p_t[:, 0], p_t[:, 1], p_t[:, 2], '-.', color='tab:pink', linewidth=1)
    ax_3d.plot(p_t[:, 0], p_t[:, 1], 0*p_t[:, 2] - .5, 'k', linewidth=1, alpha=0.25)
    ax_3d.plot(p_r[:, 0], p_r[:, 1], p_r[:, 2], 'r', linewidth=1)
    ax_3d.plot(p_r[:, 0], p_r[:, 1], 0*p_r[:, 2] - .5, 'k', linewidth=1, alpha=0.25)
    ax_3d.plot(p_target.x, p_target.y, p_target.z, 'or', markersize=2, label=r'$\mathbf{p}_t$')
    ax_3d.plot(p_target.x, p_target.y, 0*p_target.z - .5, 'ok', alpha=0.25, markersize=2)
    draw_axes(ax_3d, dq_d_[::10], .075)
    draw_axes(ax_3d, dq_f_[::20], .075)
    draw_axes(ax_3d, dq_t_[::20], .075)
    draw_axes(ax_3d, dq_r_[::20], .075)

    ax_3d.set_proj_type('ortho')
    ax_3d.set_xlim([-.5, 1.50])
    ax_3d.set_ylim([-.25, 1.75])
    ax_3d.set_zlim([-.5, 1.50])
    ax_3d.set_xlabel(r'$x$', fontsize=12)
    ax_3d.set_ylabel(r'$y$', fontsize=12)
    ax_3d.set_zlabel(r'$z$', fontsize=12)
    ax_3d.xaxis.labelpad = -5
    ax_3d.yaxis.labelpad = -5
    ax_3d.zaxis.labelpad = -5
    ax_3d.xaxis._axinfo["grid"]['linewidth'] = .1
    ax_3d.yaxis._axinfo["grid"]['linewidth'] = .1
    ax_3d.zaxis._axinfo["grid"]['linewidth'] = .1
    ax_3d.tick_params(axis='both', which='major', labelsize=5, pad=-2, grid_alpha=1)
    ax_3d.view_init(elev=30, azim=45)
    ax_3d.set_box_aspect((1, 1, 1), zoom=.90)
    fig_3d.legend(loc="lower center", fontsize=6)

    plt.show()

    fig_3d.savefig("./src/figures/figure_baxterdemo3d.png", dpi=200, bbox_inches="tight")


if __name__ == '__main__':
    main()
