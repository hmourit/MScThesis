from __future__ import division, print_function

import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import LeaveOneOut
from data import load_mdd_data
from time import time

if __name__ == '__main__':
    data, factors = load_mdd_data()

    alpha_range = np.logspace(-2, 7, 10)
    l1_ratio_range = np.arange(0., 1., 10)
    param_grid = dict(alpha=alpha_range, l1_ratio=l1_ratio_range)
    clf = SGDClassifier(loss='log', penalty='elasticnet')
    cv = LeaveOneOut(data.shape[0])
    grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, n_jobs=-1)

    t0 = time()
    grid.fit(data, factors['stress'])
    print('Training time: {}'.format(time() - t0))
    score = grid.score(data, factors['stress'])
    print('Training accuracy: {}'.format(score))
