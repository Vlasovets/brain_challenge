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


# ### Single GL plus low-rank
lambda1_range = np.logspace(-0.9, -1.5, 4)
mu1_range = np.arange(6.25, 11,  0.5)[::-1]

K = len(corr)
N = K*[corr.shape[1]]


start_time = datetime.now()

low_est_uniform, low_est_indv, low_statistics = K_single_grid(corr, lambda1_range, N,
                                                  mu_range = mu1_range,
                                                  method = 'eBIC', gamma = 0.3, 
                                                  latent = True, use_block = True)

end_time = datetime.now()

run_time = end_time - start_time

low_statistics['time'] = run_time
print("--- TIME: {0} ---".format(run_time))


if not os.path.exists(storage_dir + "/low_est_uniform/"):
    os.makedirs(storage_dir + "/low_est_uniform/")
    
if not os.path.exists(storage_dir + "/low_est_individ/"):
    os.makedirs(storage_dir + "/low_est_individ/")


ix = 0
# dump matrices into csv
for i in batch_1:
    np.savetxt(storage_dir + "/low_est_uniform/low_est_uniform{0}.csv".format(i), low_est_uniform["Theta"][ix], 
               delimiter=",", header='')
    np.savetxt(storage_dir + "/low_est_individ/low_est_individ{0}.csv".format(i), low_est_uniform["L"][ix], 
               delimiter=",", header='')
    
    
    np.savetxt(storage_dir + "/low_est_uniform/low_est_uniform{0}.csv".format(i), low_est_indv["Theta"][ix], 
               delimiter=",", header='')
    np.savetxt(storage_dir + "/low_est_individ/low_est_individ{0}.csv".format(i), low_est_indv["L"][ix], 
               delimiter=",", header='')
    
    
    ix += 1
    

with open('low_statistics_SGL_0_300.txt', 'w') as f:
    print(low_statistics, file=f)