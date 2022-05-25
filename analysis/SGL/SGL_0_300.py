#!/usr/bin/env python
# coding: utf-8

import gglasso
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

from numpy import genfromtxt
from matplotlib.pyplot import figure
from datetime import datetime
from gglasso.helper.model_selection import K_single_grid


# ### Read data

corr_all_ix = np.arange(0, 950)
outliers_ix = [96, 144, 210, 522]

corr_filtered_ix = np.array([i for i in corr_all_ix if i not in outliers_ix])
batch_1 = corr_filtered_ix[:300]

storage_dir = "/lustre/groups/bds01/datasets/brains/"

corr = []

for i in batch_1:
    corr.append(genfromtxt(storage_dir + "corr_matrices/corr{0}.csv".format(i), delimiter=','))

corr = np.array(corr)


# ### Single GL
lambda1_range = np.logspace(-0.9, -1.5, 10)

K = len(corr)
N = K*[corr.shape[1]]

start_time = datetime.now()

est_uniform, est_indv, statistics = K_single_grid(corr, lambda1_range, N, 
                                                  method = 'eBIC', gamma = 0.3, 
                                                  latent = False, use_block = True)


end_time = datetime.now()

run_time = end_time - start_time

statistics['time'] = run_time
print("--- TIME: {0} ---".format(run_time))


if not os.path.exists(storage_dir + "/est_uniform/"):
    os.makedirs(storage_dir + "/est_uniform/")
    
if not os.path.exists(storage_dir + "/est_individ/"):
    os.makedirs(storage_dir + "/est_individ/")


ix = 0
# dump matrices into csv
for i in batch_1:
    np.savetxt(storage_dir + "/est_uniform/est_uniform{0}.csv".format(i), est_uniform["Theta"][ix], 
               delimiter=",", header='')
    np.savetxt(storage_dir + "/est_individ/est_individ{0}.csv".format(i), est_indv["Theta"][ix], 
               delimiter=",", header='')
    ix += 1
    

with open('statistics_SGL_0_300.txt', 'w') as f:
    print(statistics, file=f)

