
from __future__ import division, print_function

from utils import Timer, load_data, log_results

import numpy as np

from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from datetime import datetime
from joblib import Parallel, delayed
from pprint import pprint
import sys
from sklearn.linear_model import SGDClassifier

# #### Simplified

# In[1]:

rma, drug, stress = load_data()

from sklearn.cross_validation import cross_val_score

alpha_range = np.logspace(-2, 7, 10)
l1_ratio_range = np.arange(0., 1., 0.1)
param_grid = dict(alpha=alpha_range, l1_ratio=l1_ratio_range)

n_iter = 30
n_folds = 10
test_size = 0.1

if __name__ == '__main__':

    test_size = float(sys.argv[1])
    n_iter = int(sys.argv[2])
    n_folds = int(sys.argv[3])
    target = sys.argv[4]

    if target == 'drug':
        target = drug
    else:
        target = stress

    timer = Timer()
    print('Starting...')
    split = StratifiedShuffleSplit(target['str'], n_iter=n_iter, test_size=test_size)
    clf = SGDClassifier(loss='log', penalty='elasticnet', n_jobs=1)
    grid = GridSearchCV(clf, param_grid=param_grid, cv=n_folds, n_jobs=1)

    def foo(grid, train_data, target, train_idx):
        grid.fit(rma.iloc[train_idx], target.ix[train_idx, 'str'])
        return grid.best_params_

    params = Parallel(n_jobs=n_iter, verbose=11)(delayed(foo)(grid, rma, target, train_idx) for train_idx, _ in split)

    print(timer.elapsed())
    pprint(params)

