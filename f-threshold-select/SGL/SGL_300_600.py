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
start = 300
stop = 600

corr = []

for i in range(start, stop):
    corr.append(genfromtxt("/storage/groups/bds01/datasets/brains/corr_matrices/corr{0}.csv".format(i), delimiter=','))

corr = np.array(corr)
corr.shape


# ### Single GL
# lambda1_range = np.logspace(0, -3, 8)
lambda1_range = np.logspace(-0.9, -1.5, 10)

N = corr.shape[1]


est_uniform, est_indv, statistics = K_single_grid(corr, lambda1_range, N, 
                                                  method = 'eBIC', gamma = 0.3, 
                                                  latent = False, use_block = True)


K = "301-600"


os.mkdir("/storage/groups/bds01/datasets/brains/est_uniform{0}/".format(K))
os.mkdir("/storage/groups/bds01/datasets/brains/est_individ{0}/".format(K))

ix=0
# dump matrices into csv
for i in range(start, stop):
    np.savetxt("/storage/groups/bds01/datasets/brains/est_uniform{0}/est_uniform{1}.csv".format(K, i), est_uniform["Theta"][ix], 
               delimiter=",", header='')
    np.savetxt("/storage/groups/bds01/datasets/brains/est_individ{0}/est_individ{1}.csv".format(K, i), est_indv["Theta"][ix], 
               delimiter=",", header='')
    ix +=1
    
with open("statistics{0}.txt".format(K), 'w') as f:
    print(statistics, file=f)