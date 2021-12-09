#!/usr/bin/env python
# coding: utf-8

import gglasso
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt

from numpy.linalg import matrix_rank
from matplotlib.pyplot import figure
from scipy import stats
from scipy.linalg import eigh
from numpy import genfromtxt

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

sub_corr = []

for i in range(0, 50):
    sub_corr.append(genfromtxt("/storage/groups/bds01/datasets/brains/sub_corr50/sub_corr{0}.csv".format(i), delimiter=','))

sub_corr = np.array(sub_corr)
sub_corr.shape


# ### Single GL

lambda1_range = np.logspace(0, -3, 8)
mu1_range = np.logspace(0, -1, 5)


N = sub_corr.shape[1]


est_uniform, est_indv, statistics = K_single_grid(sub_corr, lambda1_range, N,
                                                  mu_range = mu1_range,
                                                  method = 'eBIC', gamma = 0.3, 
                                                  latent = True, use_block = True)


K = est_uniform["Theta"].shape[0]


# dump matrices into csv
for i in range(0, K):
    np.savetxt("/storage/groups/bds01/datasets/brains/est_uniform_latent50/Theta{0}.csv".format(i), est_uniform["Theta"][i], 
               delimiter=",", header='')
    np.savetxt("/storage/groups/bds01/datasets/brains/est_uniform_latent50/L{0}.csv".format(i), est_uniform["L"][i], 
               delimiter=",", header='')




with open('/storage/groups/bds01/datasets/brains/statistics_SGL_latent50.txt', 'w') as f:
    print(statistics, file=f)

