#!/usr/bin/env python
# coding: utf-8

# %cd f-threshold-select


# !git clone https://github.com/fabian-sp/GGLasso.git


import gglasso
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import os
import matplotlib.pyplot as plt

from numpy.linalg import matrix_rank
from matplotlib.pyplot import figure
from scipy import stats
from scipy.linalg import eigh
from numpy import genfromtxt
from datetime import datetime
import time

from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.problem import glasso_problem

from gglasso.helper.data_generation import generate_precision_matrix, group_power_network, sample_covariance_matrix
from gglasso.helper.basic_linalg import adjacency_matrix
from gglasso.helper.data_generation import time_varying_power_network, sample_covariance_matrix
from gglasso.helper.experiment_helper import lambda_grid, discovery_rate, error
from gglasso.helper.utils import get_K_identity
from gglasso.helper.experiment_helper import plot_evolution, plot_deviation, surface_plot, single_heatmap_animation
from gglasso.helper.model_selection import aic, ebic, K_single_grid


# ### Read data

start = 600
stop = 951

corr = []

for i in range(start, stop):
    corr.append(genfromtxt("/storage/groups/bds01/datasets/brains/corr_matrices/corr{0}.csv".format(i), delimiter=','))

corr = np.array(corr)
corr.shape


# ### Single GL plus low-rank
lambda1_range = np.logspace(-0.9, -1.5, 4)
mu1_range = np.arange(6.25, 11,  0.5)[::-1]

N = corr.shape[1]


start_time = datetime.now()

low_est_uniform, low_est_indv, low_statistics = K_single_grid(corr, lambda1_range, N,
                                                  mu_range = mu1_range,
                                                  method = 'eBIC', gamma = 0.3, 
                                                  latent = True, use_block = True)

end_time = datetime.now()

run_time = end_time - start_time

low_statistics['time'] = run_time
print("--- TIME: {0} ---".format(run_time))

K = "600-951"


os.mkdir("/storage/groups/bds01/datasets/brains/low_est_uniform{0}/".format(K))
os.mkdir("/storage/groups/bds01/datasets/brains/low_est_individ{0}/".format(K))


ix=0
# dump matrices into csv
for i in range(start, stop):
    np.savetxt("/storage/groups/bds01/datasets/brains/low_est_uniform{0}/low_est_uniform_Theta{1}.csv".format(K, i), low_est_uniform["Theta"][ix], 
               delimiter=",", header='')
    np.savetxt("/storage/groups/bds01/datasets/brains/low_est_uniform{0}/low_est_uniform_L{1}.csv".format(K, i), low_est_uniform["L"][ix], 
               delimiter=",", header='')
    
    
    np.savetxt("/storage/groups/bds01/datasets/brains/low_est_individ{0}/low_est_individ_Theta{1}.csv".format(K, i), low_est_indv["Theta"][ix], 
               delimiter=",", header='')
    np.savetxt("/storage/groups/bds01/datasets/brains/low_est_individ{0}/low_est_individ_L{1}.csv".format(K, i), low_est_indv["L"][ix], 
               delimiter=",", header='')
    ix +=1
    
with open("low_statistics{0}.txt".format(K), 'wb') as handle:
    pickle.dump(low_statistics, handle, protocol=pickle.HIGHEST_PROTOCOL)