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
from datetime import datetime

from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.problem import glasso_problem

from gglasso.helper.data_generation import generate_precision_matrix, group_power_network, sample_covariance_matrix
from gglasso.helper.basic_linalg import adjacency_matrix
from gglasso.helper.data_generation import time_varying_power_network, sample_covariance_matrix
from gglasso.helper.experiment_helper import lambda_grid, discovery_rate, error
from gglasso.helper.utils import get_K_identity
from gglasso.helper.experiment_helper import plot_evolution, plot_deviation, surface_plot, single_heatmap_animation
from gglasso.helper.model_selection import aic, ebic


df = pd.read_csv("/storage/groups/bds01/datasets/brains/merged.csv", sep=',')


sex = df["sex_f0_m1"]
age = df["age"]

df = df.iloc[:, :-2]
print(df.shape)


#create a diagonal matrix with ones
ix_list = []
range_ = np.arange(0,436)
sum_ = sum(range_)

#list of indices where should be 1
for i in range(0, 436):
    ix = sum_ - sum(range(0, 436 - i)) + i
    ix_list.append(ix)
    
#create an empty list of the shape of data
empty = np.empty([951, 95266], dtype=float)


for i, row in df.iterrows():
    a = np.array(row)
    
    for ix in ix_list:
        a = np.insert(a, ix, 1, axis=0)
    
    empty[i, :] = a
    

diag_df = pd.DataFrame(empty)
print(diag_df.shape)


#from vectors to matrices
corr = []
for _, i in diag_df.iterrows():
    #split a row to subrows representing columns in corr matrix
    #exclude 0 index from splitting criteria
    items = np.split(i, ix_list[1:])
    
    # get the maximum  col length
    maxlen = len(max(items, key=len))
    
    # pad left of each sublist with NaN to make it as long as the longest
    i = 0
    for item in items:
        nans = np.repeat(np.nan, (maxlen - len(item)))
        item = np.concatenate((nans, item), axis=None)
        items[i] = item
        i += 1
        
    #create upper and lower triangular matrices
    upper = pd.DataFrame.from_records(items)
    lower = upper.T
    
    #full corr matrix
    c = upper.fillna(lower)
    
    corr.append(c)
    
    
C = np.array(corr)
print(C.shape)


ein_counts = []

for i in range(0, C.shape[0]):
    for j in range(0, C.shape[1]):
        if C[i][j].any() >= 1:
            #print("Not a corr value at matrix - {0}".format(i))
            ein_counts.append(1)
        elif C[i][j].any() <= -1:
            #print("Not a corr value at {0}".format(i))
            ein_counts.append(1)
print("If 1 is only at the diagonal: 951*436 = {0}".format(951*436))
print("Total number of ones: {0}".format(np.array(ein_counts).sum()))


#check if any NaN
print(pd.DataFrame(C[0]).isnull().values.any())




# (436,436) identity matrix
I = np.eye(C.shape[1])

tol_up = 10e-6
tol_low = 10e-14

corr_mod = []

# 951 matrices
for i in range(0, C.shape[0]):
    matrix = C[i]
    
    eigenvalues = eigh(matrix, eigvals_only=True)
    #change negative to zero first and then continue - does not make sense becuase the reproducible matrix will have neagtive eigenvalues again.
    min_positive = np.select(eigenvalues > 0, eigenvalues)
    
    if min_positive > tol_up:
        min_positive = tol_up
    elif min_positive < tol_low:
        min_positive = tol_low
    
    matrix = matrix + I*min_positive
    
    try:
        np.linalg.cholesky(matrix)
        corr_mod.append(matrix)
    except:
        print("Some matrices are not SPD")
        
        
corr_mod = np.array(corr_mod)
print(corr_mod.shape)


# #dump matrices into csv
for i in range(0, corr_mod.shape[0]):
    np.savetxt("/storage/groups/bds01/datasets/brains/corr_matrices/corr{0}.csv".format(i), corr_mod[i], delimiter=",", header='')
