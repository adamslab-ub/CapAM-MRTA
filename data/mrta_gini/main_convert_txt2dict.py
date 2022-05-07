# Author: Payam Ghassemi, payamgha@buffalo.edu
# Dec 02, 2020
# Copyright 2020 Payam Ghassemi

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from scipy.spatial import distance as dist

import scipy.io

import pickle

## TAPTC Dataset
group_list = [1,2]
instance_list = [0, 1, 2]
ratio_deadline_list = [1, 2, 3, 4]
robotSize_list = [2, 3, 5, 7]

pkl_file_names = []
all_files = "gini_data_sets_n_100.pkl"

for G in group_list:
    for D in ratio_deadline_list:
        for R in robotSize_list:
            for I in instance_list:
                agent_name = "a"+str(R)+"i0"+str(I)
                data_name = "r"+str(G)+str(D)+agent_name
                dir_name = "group"+str(G)
                file_name = dir_name+"/"+data_name+".txt"
                tasks = pd.read_csv(file_name, sep=" ", header=None, skiprows=1)
                tasks.columns = ["id", "x", "y", "w", "T"]
                file_name = "agent/"+agent_name+".txt"
                robots = pd.read_csv(file_name, sep=" ", header=None, skiprows=1)
                robots.columns = ["id", "x", "y", "c"]
                loc = tasks.values[:, 1:3]/100
                workload = tasks.values[:,3]/100
                deadline = tasks.values[:,4]
                loc_data = {
                    'loc' : loc,
                    'workload': workload,
                    'deadline':deadline,
                    'n_agents':R,
                }



                robots_loc = robots.values[:,1:3]/100
                robots_capacity = robots.values[:, 3]

                robot_data = {
                    'robots_loc': robots_loc,
                    'robots_capacity': robots_capacity / 100
                }

                data = {
                    'loc_data': loc_data,
                    'robot_data': robot_data
                }

                pkl_file_name = data_name+".pkl"

                pkl_file_names.append("data/mrta_gini/" + pkl_file_name)

                pickle.dump(data, open(pkl_file_name, "wb"))

                print(pkl_file_name)


file_n = open(all_files, 'wb')
pickle.dump(pkl_file_names, file_n)
file_n.close()

file_n = open(all_files, 'rb')
dt = pickle.load(file_n)
ft = 0

