
# coding: utf-8

# In[22]:

from __future__ import division, print_function
import numpy as np
import os
import sys
import time


# In[56]:

RESULTS_FILE = '../results'
def log_results(log):
    with open(RESULTS_FILE, 'r') as in_:
        logs = json.load(in_)
    with open(RESULTS_FILE, 'w') as out:
        logs.append(log)
        json.dump(logs, out, sort_keys=True, indent=2, separators=(',', ': '))


# In[68]:

class Timer():
    def __init__(self):
        self.start = datetime.now()
        
    def elapsed(self):
        return str(datetime.now() - self.start)[:-7]


# ## Load data

# In[71]:

import cPickle as pickle
with open('rma.pickle', 'rb') as in_:
    rma = pickle.load(in_)
with open('drug.pickle', 'rb') as in_:
    drug = pickle.load(in_)
with open('stress.pickle', 'rb') as in_:
    stress = pickle.load(in_)


# ## Experiments

# In[23]:

from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from datetime import datetime


# In[75]:

C_range = np.logspace(-2, 7, 10)
gamma_range = np.logspace(-6, 3, 10)
param_grid = dict(gamma=gamma_range, C=C_range)

n_iter = 10
n_folds = 10
test_size = 0.3

log = {
    'target': 'stress',
    'split': {
        'type': 'StratifiedShuffleSplit',
        'n_iter': n_iter,
        'test_size': test_size
    },
    'cross_val': {
        'type': 'StratifiedKFold',
        'n_folds': n_folds,
        'shuffle': True
    },
    'classifier': 'SVC'
    
}

accuracy = np.full((n_iter,), np.nan)

timer = Timer()
split = StratifiedShuffleSplit(stress['str'], n_iter=n_iter, test_size=test_size)
for i, (train, test) in enumerate(split):
    print('{}: Iteration {}'.format(timer.elapsed(), i), end='')
    
    cross_val = StratifiedKFold(stress.ix[train, 'str'], n_folds=n_folds, shuffle=True)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cross_val, n_jobs=-1)
    grid.fit(rma.iloc[train], stress.ix[train, 'str'])
    
    pred = grid.predict(rma.iloc[test])
    accuracy[i] = accuracy_score(stress.ix[test, 'str'], pred)

    print('\tAccuracy: {:.2%} +/- {:.2%}'.format(np.nanmean(accuracy), np.nanstd(accuracy)))
    

print('\n{}: Accuracy: {:.2%} +/- {:.2%}'.format(timer.elapsed(), np.nanmean(accuracy), np.nanstd(accuracy)))

log['results'] = {'accuracy': {'mean': accuracy.mean(), 'std': accuracy.std()}, 'time': timer.elapsed()}
log_results(log)

