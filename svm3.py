
from __future__ import division, print_function

from utils import Timer, load_data, log_results

import numpy as np

from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from datetime import datetime


# #### Simplified

# In[1]:

rma, drug, stress = load_data()

from sklearn.cross_validation import cross_val_score

C_range = np.logspace(-2, 7, 10)
gamma_range = np.logspace(-6, 3, 10)
param_grid = dict(gamma=gamma_range, C=C_range)

n_iter = 30
n_folds = 10
test_size = 0.1

log = {
    'target': 'drug',
    'split': {
        'type': 'StratifiedShuffleSplit',
        'n_iter': n_iter,
        'test_size': test_size
    },
    'cross_val': {'n_folds': n_folds},
    'classifier': 'SVC'
    
}

timer = Timer()
print('Starting...')
split = StratifiedShuffleSplit(drug['str'], n_iter=n_iter, test_size=test_size)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=n_folds, n_jobs=1)
accuracy = cross_val_score(grid, rma, y=drug['str'], scoring='accuracy', cv=split, n_jobs=n_iter, verbose=1)
print('\n{}: Accuracy: {:.2%} +/- {:.2%}'.format(timer.elapsed(), np.nanmean(accuracy), np.nanstd(accuracy)))

log['results'] = {'accuracy': {'mean': accuracy.mean(), 'std': accuracy.std()}, 'time': timer.elapsed()}
log_results(log)
