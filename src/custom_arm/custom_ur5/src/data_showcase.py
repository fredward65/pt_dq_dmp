#!/usr/bin/env python3

import numpy as np
import glob
import pickle
from rospkg import RosPack

np.set_printoptions(precision=3, suppress=True)


def main():
    file_path = RosPack().get_path("custom_ur5") + "/resources/"
    file_list = glob.glob(file_path + "*.pkl")
    
    for file_name in file_list:
        with open(file_name, "rb") as handle:
            data_dict = pickle.load(handle)
            if (data_dict['repetitions'] == 100):
                t_pos = data_dict['target']
                print("Original Target Pose: ", t_pos)
                planner_list = ['ProjEST', 'PDST', 'RRTstar', 'LazyPRMstar']
                for planner in planner_list:
                    b_pos = data_dict[planner]['results']
                    p_mean = np.mean(b_pos, axis=0)
                    p_err = b_pos - t_pos
                    p_err_mean = np.mean(p_err, axis=0)
                    p_err_norm_mean = np.mean(np.linalg.norm(p_err, axis=1))
                    p_err_norm_stdd = np.std(np.linalg.norm(b_pos - t_pos, axis=1))
                    p_err_min_norm = np.min(np.linalg.norm(p_err, axis=1))
                    print("Planner: ", planner)
                    print("Mean Ball Pose: ", p_mean)
                    print("Mean Ball Pose Difference: ", p_err_mean)
                    print("Mean Ball Pose Difference Norm: ", p_err_norm_mean)
                    print("Mean Ball Pose Difference Std. Dev.: ", p_err_norm_stdd)
                    print("Min Ball Poses Difference Norm: ", p_err_min_norm)


if __name__ == "__main__":
    main()
