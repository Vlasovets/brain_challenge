import os
import gglasso
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from xgboost import XGBClassifier


from matplotlib.pyplot import figure

from numpy.linalg import matrix_rank
from numpy import genfromtxt

from scipy import stats
from scipy.linalg import eigh

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score


corr_all_ix = np.arange(0, 950)
outliers_ix = [96, 144, 210, 522]

corr_filtered_ix = np.array([i for i in corr_all_ix if i not in outliers_ix])
corr_filtered_ix.shape

storage_dir = "/lustre/groups/bds01/datasets/brains/"

outliers = []

for i in outliers_ix:
    outliers.append(genfromtxt(storage_dir + "corr_matrices/corr{0}.csv".format(i), delimiter=','))

    
outliers = np.array(outliers)
outliers.shape


corr = []
sol = []

for i in corr_filtered_ix:
    corr.append(genfromtxt(storage_dir + "/corr_matrices/corr{0}.csv".format(i), delimiter=','))
    sol.append(genfromtxt(storage_dir + "/est_uniform/est_uniform{0}.csv".format(i), delimiter=','))
    
sol = np.array(sol)
corr = np.array(corr)
corr.shape, sol.shape


sex = pd.read_csv(storage_dir + "sex.csv")
age = pd.read_csv(storage_dir + "age.csv")

#remove outliers
sex = sex.iloc[corr_filtered_ix]
age = age.iloc[corr_filtered_ix]


X, y = sol, sex
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

n_train = X_train.shape[0]
n_test = X_test.shape[0]

X_train = X_train.reshape(n_train, X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(n_test, X_test.shape[1]*X_test.shape[2])





#adjust CV threshold so you don't have only one class in a sample
heldout = [0.3, 0.2, 0.1]
# Number of rounds to fit and evaluate an estimator.
rounds = 10
X, y = sol, sex

classifiers = [
     ("XGBoost", xgb.XGBClassifier()),
    ("Log-regression", LogisticRegression(max_iter= 110, penalty='l2')),
   ("SVM", svm.SVC(kernel='linear'))
]

xx = 1.0 - np.array(heldout)

for name, clf in classifiers:
    print("training %s" % name)
    rng = np.random.RandomState(42)
    yy = []
    for i in heldout:
        yy_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=i, random_state=rng
            )
            
            n_train = X_train.shape[0]
            n_test = X_test.shape[0]

            X_train = X_train.reshape(n_train, X_train.shape[1]*X_train.shape[2])
            X_test = X_test.reshape(n_test, X_test.shape[1]*X_test.shape[2])
            
            clf.fit(X_train, y_train.values.ravel())
            y_pred = clf.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
    plt.plot(xx, yy, label=name)

plt.legend(loc="upper right")
plt.xlabel("Proportion train")
plt.ylabel("Test Error Rate")
plt.show()