#!/usr/bin/env python3

import numpy as np
from custom_tools.math_tools import *
from custom_tools.math_tools.dq_tools import edq_from_dq_list, twist_from_dq_list, next_dq_from_twist, vel_from_twist
from custom_tools.projectile_throwing import gen_movement, ProjectileThrowing
from custom_tools.pt_dq_dmp import PTDQDMP
from dual_quaternions import DualQuaternion
from matplotlib import animation, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion
from rospkg import RosPack

np.set_printoptions(precision=3, suppress=True)
plt.rcParams.update({'text.usetex': True, 'font.size': 7, 'figure.dpi': 300, 'axes.linewidth': 0.25})
plt.style.use('dark_background')

file_path = RosPack().get_path("custom_ur5") + "/resources/"


def draw_axes(ax, dq_list):
    for dq_i in dq_list:
        p_i = 2 * dq_log(dq_i).q_d
        q_i = dq_i.q_r
        v_x = q_i.rotate([1, 0, 0])
        v_y = q_i.rotate([0, 1, 0])
        v_z = q_i.rotate([0, 0, 1])
        ax.quiver(p_i.x, p_i.y, p_i.z, v_x[0], v_x[1], v_x[2], length=0.1, linewidth=.5, colors=[1,0,0,1])
        ax.quiver(p_i.x, p_i.y, p_i.z, v_y[0], v_y[1], v_y[2], length=0.1, linewidth=.5, colors=[0,1,0,1])
        ax.quiver(p_i.x, p_i.y, p_i.z, v_z[0], v_z[1], v_z[2], length=0.1, linewidth=.5, colors=[0,0,1,1])


def dq_animate(t_vec, dq_arr, title=None, filename='default', color='c'):
    dt = 1000*np.mean(np.diff(t_vec))

    fig = plt.figure(figsize=(1, 4))
    # fig.suptitle(title)
    axs = fig.subplots(8, 1)
    lines = []
    for i, ax in enumerate(axs):
        ax.set_xlim(t_vec[0], t_vec[-1])
        ax.set_ylim(np.floor(np.min(dq_arr[:, i]))-.1, np.ceil(np.max(dq_arr[:, i]))+.1)
        ax.set_xticks([])
        ax.set_yticks([])
        line = ax.plot(t_vec[0].reshape((-1, 1)), dq_arr[0, i], color=color, linewidth=.75)[0]
        lines.append(line)
    fig.tight_layout()

    def update(frame):
        for i, line in enumerate(lines):
            line.set_xdata(t_vec[:frame])
            line.set_ydata(dq_arr[:frame, i])
            
    ani = animation.FuncAnimation(fig=fig, func=update, frames=t_vec.shape[0], interval=dt, repeat=False)
    ani.save(filename="%s.avi" % filename, writer="ffmpeg")
    # plt.show()


def dq_dmp():
    # t_vec, dq_vec = gen_movement(r=.40, n=500)   # .35
    # p_0 = Quaternion(vector=[-.65, -.05, .40])    # -.35 .35 .45 
    # q_0 = Quaternion(axis=[0, 0, 1], angle=0.00 * np.pi)
    # dq_0 = DualQuaternion.from_quat_pose_array(np.append(q_0.elements, p_0.elements[1:]))
    # dq_vec = np.array([dq_0 * dq_i for dq_i in dq_vec], dtype=DualQuaternion)
    file_name = "demo_throw_left_4.csv"
    data = np.loadtxt(file_path + file_name, delimiter=',', skiprows=1)
    t_vec = data[:, 0]
    p_vec = data[:, 1:4]
    q_vec = np.c_[data[:, -1], data[:, 4:-1]]
    dq_vec = np.array([DualQuaternion.from_quat_pose_array(np.append(q_i, p_i)) for q_i, p_i in zip(q_vec, p_vec)])

    tg = t_vec[-1] - t_vec[0]
    n = 20
    tau = 1
    alpha_xi = 12
    beta_xi = alpha_xi / 4
    alpha_x = (alpha_xi / 3) * (1 / tg)

    edq_vec = edq_from_dq_list(dq_vec)
    tw_vec = twist_from_dq_list(t_vec, dq_vec)
    dtw_vec = dx_dt(t_vec, tw_vec)

    fd_vec = dtw_vec + (-1 * alpha_xi * ((beta_xi * edq_vec) + (-1 * tw_vec)))
    fd_arr = dql_to_npa(fd_vec)

    x = np.exp(-alpha_x * t_vec / tau).reshape((-1, 1))
    c_i = np.exp(-alpha_x * ((np.linspace(1, n, n) - 1) / (n - 1)) * tg)
    h_i = n / np.power(c_i, 2)

    psi_i = np.empty([len(x), n], dtype=float)
    for i in range(n):
        psi_i[:, i] = np.exp(-1 * (h_i[i] * np.power(x - c_i[i], 2))).reshape(-1)

    w_i = np.empty([fd_arr.shape[1], n])
    for i in range(n):
        psi_m = np.diag(psi_i[:, i])
        w_i[:, i] = np.dot(np.dot(x.T, psi_m), fd_arr) / np.dot(np.dot(x.T, psi_m), x)

    tn_vec = 2 * t_vec
    xn = np.exp(-alpha_x * tn_vec / tau).reshape((-1, 1))
    psi_n = np.empty([len(xn), n], dtype=float)
    for i in range(n):
        psi_n[:, i] = np.exp(-1 * (h_i[i] * np.power(xn - c_i[i], 2))).reshape(-1)

    fn_arr = ((xn.reshape((-1, 1)) * np.inner(psi_n, w_i)) / np.sum(psi_n, 1).reshape((-1, 1))).reshape((-1, 8))
    fn_vec = [DualQuaternion.from_dq_array(fn_i) for fn_i in fn_arr]

    dqn_vec = np.empty(t_vec.shape[0], dtype=DualQuaternion)
    edqn_vec = np.empty(t_vec.shape[0], dtype=DualQuaternion)
    twn_vec = np.empty(t_vec.shape[0], dtype=DualQuaternion)
    dtwn_vec = np.empty(t_vec.shape[0], dtype=DualQuaternion)

    dqg = dq_vec[-1]
    dq_i = dq_vec[0]
    tw_i = tw_vec[0]
    t_p = tn_vec[0]
    for i, (t_i, f_i) in enumerate(zip(tn_vec, fn_vec)):
        dt = t_i - t_p
        edq = edq_from_dq(dq_i, dqg)
        dtw = (1 / tau) * (f_i + alpha_xi * ((beta_xi * edq) + (-1 * tw_i)))
        dqn_vec[i] = dq_i
        twn_vec[i] = dq_i
        edqn_vec[i] = edq
        dtwn_vec[i] = dtw
        tw_i = tw_i + ((dt / tau) * dtw)
        dq_i = next_dq_from_twist(dt, dq_i, tw_i)
        t_p = t_i
    
    """ Demonstrated Variables """
    # dq_arr = dql_to_npa(dq_vec)
    # tw_arr = dql_to_npa(tw_vec)
    # edq_arr = dql_to_npa(edq_vec)
    # dtw_arr = dql_to_npa(dtw_vec)
    # dq_animate(t_vec, dq_arr, r'$\underline{\mathbf{q}}_\mathrm{d}$', file_path + "dq_elements")
    # dq_animate(t_vec, edq_arr, r'$\mathrm{e}_{{\underline{\mathbf{q}}_\mathrm{g}}_\mathrm{d}}$', file_path + "edq_elements")
    # dq_animate(t_vec, tw_arr, r'$\underline{\xi}_{b_\mathbf{d}}$', file_path + "tw_elements")
    # dq_animate(t_vec, dtw_arr, r'$\dot{\underline{\xi}}_{b_\mathbf{d}}$', file_path + "dtw_elements")
    # dq_animate(t_vec, fd_arr, r'$\mathbf{f}_\mathrm{d}$', file_path + "fd_elements")

    """ Reconstructed Variables """
    # dqn_arr = dql_to_npa(dqn_vec)
    # edqn_arr = dql_to_npa(edqn_vec)
    # twn_arr = dql_to_npa(twn_vec)
    # dtwn_arr = dql_to_npa(dtwn_vec)
    # dq_animate(t_vec, dqn_arr, r'$\underline{\mathbf{q}}_\mathrm{f}$', file_path + "dqn_elements", 'm')
    # dq_animate(t_vec, edqn_arr, r'$\mathrm{e}_{{\underline{\mathbf{q}}_\mathrm{g}}_\mathrm{f}}$', file_path + "edqn_elements", 'm')
    # dq_animate(t_vec, twn_arr, r'$\underline{\xi}_{b_\mathbf{f}}$', file_path + "twn_elements", 'm')
    # dq_animate(t_vec, dtwn_arr, r'$\dot{\underline{\xi}}_{b_\mathbf{f}}$', file_path + "dtwn_elements", 'm')
    # dq_animate(t_vec, fn_arr, r'$\mathbf{f}_\mathrm{f}$', file_path + "fn_elements", 'm')

    """ Canonical System """
    # fig = plt.figure(figsize=(2, 1))
    # axs = fig.add_subplot()
    # axs.set_xlim(t_vec[0], t_vec[-1])
    # axs.set_ylim(0, 1)
    # axs.set_xticks([])
    # axs.set_yticks([])
    # lines = axs.plot(t_vec[0].reshape((-1, 1)), x[0].reshape((-1, 1)), linewidth=.75)
    # axs.set_aspect('equal')
    # fig.tight_layout()

    # def update(frame):
    #     for i, line in enumerate(lines):
    #         line.set_xdata(t_vec[:frame])
    #         line.set_ydata(x[:frame])
        
    # dt = 1000*np.mean(np.diff(t_vec))
    # ani = animation.FuncAnimation(fig=fig, func=update, frames=t_vec.shape[0], interval=dt, repeat=False)
    # ani.save(filename="%s.avi" % (file_path+"x"), writer="ffmpeg")
    # plt.show()

    """ Gaussian Kernels """
    # fig = plt.figure(figsize=(2, 1))
    # axs = fig.add_subplot()
    # axs.set_xlim(t_vec[0], t_vec[-1])
    # axs.set_ylim(0, 1)
    # axs.set_xticks([])
    # axs.set_yticks([])
    # lines = axs.plot(t_vec[0].reshape((-1, 1)), psi_i[0, :].reshape((-1, n)), linewidth=.75)
    # axs.set_aspect('equal')
    # fig.tight_layout()

    # def update(frame):
    #     for i, line in enumerate(lines):
    #         line.set_xdata(t_vec[:frame])
    #         line.set_ydata(psi_i[:frame, i])
        
    # dt = 1000*np.mean(np.diff(t_vec))
    # ani = animation.FuncAnimation(fig=fig, func=update, frames=t_vec.shape[0], interval=dt, repeat=False)
    # ani.save(filename="%s.avi" % (file_path+"psi_i"), writer="ffmpeg")
    # plt.show()

    """ Forcing Term Kernel Composition """
    # fig = plt.figure(figsize=(2, 4))
    # # fig.suptitle(r'$x \cdot w_i \cdot \Psi_i$')
    # axs = fig.subplots(8, 1)
    # line_arr = []
    # for ax in axs:
    #     ax.set_xlim(t_vec[0], t_vec[-1])
    #     ax.set_ylim(0, 1)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     lines = ax.plot(t_vec, psi_i, linewidth=0.75)
    #     line_arr.append(lines)
    # fig.tight_layout()

    # vid_len = 60
    
    # def update(frame):
    #     for i, (ax, lines) in enumerate(zip(axs, line_arr)):
    #         fac1 = 1 - np.exp(-10*((frame+1)/vid_len)) if frame < vid_len else 1
    #         x_ = (1 - fac1) + (x * fac1) if frame < vid_len else x
    #         w = (1 - fac1) + (w_i[i, :] * fac1)
    #         psi_x = x_ * psi_i * w
    #         ymax = np.max(psi_x) if np.max(psi_x) > 1 else 1
    #         ax.set_ylim(np.min(psi_x), ymax)
    #         for j, line in enumerate(lines):
    #             line.set_xdata(t_vec)
    #             line.set_ydata(psi_x[:, j])  
    
    # ani = animation.FuncAnimation(fig=fig, func=update, frames=vid_len+10, interval=30, repeat=False)
    # ani.save(filename="%s.avi" % (file_path+"psi_fd"), writer="ffmpeg")
    # plt.show()


def pt_dmp():
    # Generate movement data
    t_vec, dq_vec = gen_movement(r=.40, n=100)    # .35
    t_vec *= .5
    p_0 = Quaternion(vector=[-.65, -.05, .40])      # -.35 .35 .45 
    q_0 = Quaternion(axis=[0, 0, 1], angle=0.00 * np.pi)
    dq_0 = DualQuaternion.from_quat_pose_array(np.append(q_0.elements, p_0.elements[1:]))
    dq_vec = np.array([dq_0 * dq_i for dq_i in dq_vec], dtype=DualQuaternion)
    tw_vec = twist_from_dq_list(t_vec, dq_vec)

    # Set Cartesian target
    p_r = Quaternion(axis=[0, 0, 1], angle=-0.25 * np.pi)
    p_t = Quaternion(vector=[1.00, 0.00, 0.02])
    p_t = p_r.rotate(p_t)

    p_l_obj = ProjectileThrowing()

    dmp_obj = PTDQDMP(n=100, alpha_y=20)
    _, p_vec = pose_from_dq(dq_vec)
    dmp_obj.train_model(t_vec, dq_vec)

    # Projectile launching
    dq_g = dq_vec[-1]
    tw_g = tw_vec[-1]
    v_g = vel_from_twist(dq_g, dq_g * tw_g * dq_g.quaternion_conjugate())
    n_z = Quaternion([0.0, 0.0, 0.0, 1.0])
    n_g = (.5 * (v_g*n_z - n_z*v_g)).normalised

    p_g = 2 * dq_log(dq_g).q_d

    q_r = p_l_obj.estimate_plane_rotation(n_g, p_g, p_t)
    n_p = q_r.rotate(n_g)
    p_p = q_r.rotate(p_g)

    t_f, v_0 = p_l_obj.optimal_v_launch(p_p, p_t)
    p_est, t_est = p_l_obj.simulate_launch(t_f, v_0, p_p)
    p_dem, t_dem = p_l_obj.simulate_launch(t_f, v_g, p_g)

    q_v = quat_rot(q_r.conjugate.rotate(v_0), v_g)
    tau = v_g.norm / v_0.norm

    t_gf = t_vec[-1] * tau
    t_rec = np.linspace(0, t_vec[-1], num=100)

    tw_0 = DualQuaternion.from_dq_array(np.zeros(8))

    dq_off_0 = DualQuaternion.from_quat_pose_array(np.append(q_r.elements, [0, 0, 0]))
    dq_off_1 = DualQuaternion.from_quat_pose_array(np.append(q_v.elements, [0, 0, 0]))
    dq_0, dq_g = p_l_obj.correct_poses(dq_vec[0], dq_g, dq_off_0, dq_off_1)
    
    dq_rec, tw_rec = dmp_obj.fit_model(t_rec, dq_0, tw_0, dq_g, tau=tau)
    
    _, p_rec = pose_from_dq(dq_rec)
    f_idx = np.where(t_rec >= t_gf)[0][0]
    dq_f = dq_rec[f_idx]
    tw_f = tw_rec[f_idx]
    p_f = 2 * dq_log(dq_f).q_d
    v_f = vel_from_twist(dq_f, dq_f * tw_f * dq_f.quaternion_conjugate())
    p_fnl, t_fnl = p_l_obj.simulate_launch(t_f, v_f, p_f)
    err_g = (p_p - p_f).norm
    err_t = (p_est[-1] - p_fnl[-1]).norm
    err_v = (v_f - v_0).norm
    print("target : ", p_t.elements[1:])
    print("err goal = %5.5f, err target = %5.3f" % (err_g, err_t))
    print("tau = %5.3f, err vel = %5.3f" % (tau, err_v))

    p_the = p_fnl[-1]

    p_lnc = np.array([p_i.elements[1:] for p_i in p_est])
    p_fnl = np.array([p_i.elements[1:] for p_i in p_fnl])
    p_thr = np.array([p_i.elements[1:] for p_i in p_dem])

    """ 2D Plot """
    fig = plt.figure(figsize=(1.5, 3))
    axs = fig.subplots(8, 1)
    dq_varr = dql_to_npa(dq_vec)
    dq_rarr = dql_to_npa(dq_rec)
    dq_garr = dql_to_npa([dq_g])[0]
    lines = []
    for i, ax in enumerate(axs):
        x_max = np.max([t_vec, t_rec])
        y_min = np.min([dq_varr[:, i], dq_rarr[:, i]]) - .05
        y_max = np.max([dq_varr[:, i], dq_rarr[:, i]]) + .05 
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, y_max)
        line1 = ax.plot([t_vec[0]], [dq_varr[0, i]], 'w', linewidth=1)[0]
        line2 = ax.plot([t_rec[0]], [dq_rarr[0, i]], 'c', linewidth=1)[0]
        line3 = ax.hlines(dq_garr[i], 0, x_max, colors='m', linestyles='dashed', linewidth=.5)
        line4 = ax.vlines(t_gf, y_min, y_max, colors='m', linestyles='dashed', linewidth=.5)
        line3.set_alpha(0)
        line4.set_alpha(0)
        lines.append([line1, line2, line3, line4])
    fig.tight_layout()

    frame_len = len(t_vec) + len(t_rec)
    dt = np.mean(np.diff(t_vec)) * 1000
    def update(frame):
        for i, line in enumerate(lines):
            if frame < len(t_vec):
                line[0].set_xdata(t_vec[:frame])
                line[0].set_ydata(dq_varr[:frame, i])
            if frame >= len(t_vec) and frame < frame_len:
                idx = frame - len(t_vec)
                line[1].set_xdata(t_rec[:idx])
                line[1].set_ydata(dq_rarr[:idx, i])
                line[2].set_alpha(1)
                line[3].set_alpha(1)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=frame_len + 20, interval=dt, repeat=False)
    filename = file_path + "pt_dmp_elements"
    # ani.save(filename="%s.avi" % filename, writer="ffmpeg")

    plt.show()

    """ 3D Animated Plot """
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$y$', fontsize=12)
    ax.set_zlabel(r'$z$', fontsize=12)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.labelpad = -15
    ax.yaxis.labelpad = -15
    ax.zaxis.labelpad = -15
    ax.xaxis._axinfo["grid"]['linewidth'] = .1
    ax.yaxis._axinfo["grid"]['linewidth'] = .1
    ax.zaxis._axinfo["grid"]['linewidth'] = .1
    ax.xaxis.set_pane_color((1,1,1,0))
    ax.yaxis.set_pane_color((1,1,1,0))
    ax.zaxis.set_pane_color((1,1,1,0))
    ax.tick_params(axis='both', which='major', labelsize=7, pad=-2)
    ax.set_box_aspect((1, 1, 1), zoom=1.2)
    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=0, top=2)
    ax.set_proj_type('ortho')
    ax.view_init(elev=15, azim=-60)

    def draw_plane(ax, n, p, col='w'):
        p_1 = Quaternion([0, 0, 0, 1])
        q_1 = (0.3 * (p_1*n - n*p_1)).normalised
        ln1 = np.linspace(-.75, 1, num=10)
        ln2 = np.linspace(-.6, 1.4, num=10)
        ln1, ln2 = np.meshgrid(ln1, ln2)
        xx = p.x + p_1.x*ln1 + q_1.x*ln2
        yy = p.y + p_1.y*ln1 + q_1.y*ln2
        zz = p.z + p_1.z*ln1 + q_1.z*ln2
        ax.plot(xx[:, 0], yy[:, 0], zz[:, 0], color=col, alpha=.25)
        ax.plot_surface(xx, yy, zz, alpha=.5, antialiased=True, color=col, zorder=0)

    # ax.plot([0], [0], [0], '*k')
    # ax.plot(p_thr[:, 1], p_thr[:, 2], p_thr[:, 3], ':k')
    draw_plane(ax, n_g, p_g, 'w')
    draw_plane(ax, n_p, p_p, 'c')
    ax.quiver(p_g.x, p_g.y, p_g.z, n_g.x, n_g.y, n_g.z, length=0.2, colors='w', linewidth=.75)
    ax.quiver(p_p.x, p_p.y, p_p.z, n_p.x, n_p.y, n_p.z, length=0.2, colors='c', linewidth=.75)
    ax.plot(p_vec[:, 0], p_vec[:, 1], p_vec[:, 2], 'w', linewidth=1)
    ax.plot(p_rec[:, 0], p_rec[:, 1], p_rec[:, 2], 'c', linewidth=1)
    ax.plot([p_g.x], [p_g.y], [p_g.z], 'ow', markersize=2)
    ax.plot([p_p.x], [p_p.y], [p_p.z], 'oc', markersize=2)
    ax.quiver(p_g.x, p_g.y, p_g.z, v_g.x, v_g.y, v_g.z, length=0.2, colors='w', linewidth=.75)
    ax.quiver(p_p.x, p_p.y, p_p.z, v_0.x, v_0.y, v_0.z, length=0.2, colors='c', linewidth=.75)
    ax.quiver(p_f.x, p_f.y, p_f.z, v_f.x, v_f.y, v_f.z, length=0.2, colors='m', linewidth=.75)
    # ax.plot(p_thr[:, 0], p_thr[:, 1], p_thr[:, 2], ':w', linewidth=1)
    ax.plot(p_lnc[:, 0], p_lnc[:, 1], p_lnc[:, 2], ':c', linewidth=1)
    ax.plot(p_fnl[:, 0], p_fnl[:, 1], p_fnl[:, 2], ':m', linewidth=1)
    ax.plot([p_t.x], [p_t.y], [p_t.z], 'oc', markersize=2)
    ax.plot([p_the.x], [p_the.y], [p_the.z], 'om', markersize=2)
    # draw_axes(ax, dq_vec[::10])
    # draw_axes(ax, dq_rec[::10])
    
    # plt.legend(ncols=5, loc='lower center', bbox_to_anchor=(.5, 1.), fontsize=6)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0)

    frame_len = 100
    def update(frame):
        idx = frame + 1
        ax.view_init(elev=15, azim=-60 - 60*idx/frame_len)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=frame_len, interval=20, repeat=False)
    filename = file_path + "pt_dmp_animation"
    ani.save(filename="%s.avi" % filename, writer="ffmpeg")
    plt.show()


def main():
    # dq_dmp()
    pt_dmp()


if __name__ == "__main__":
    main()
