import numpy as np

from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from datetime import datetime
import sys
from pprint import pprint
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score
from utils_2 import load_data, log_results
from utils_2 import Timer

rma, drug, stress = load_data()

alpha_range = np.logspace(-2, 7, 10)
l1_ratio_range = np.arange(0., 1., 0.1)
en_param_grid = dict(alpha=alpha_range, l1_ratio=l1_ratio_range)

C_range = np.logspace(-2, 7, 10)
gamma_range = np.logspace(-6, 3, 10)
svm_param_grid = dict(gamma=gamma_range, C=C_range)

if __name__ == '__main__':

    test_size = float(sys.argv[1])
    n_iter = int(sys.argv[2])
    n_folds = int(sys.argv[3])
    target = sys.argv[4]
    classifier = sys.argv[5]

    log = {
        'target': target,
        'split': {
            'type': 'StratifiedShuffleSplit',
            'n_iter': n_iter,
            'test_size': test_size
        },
        'cross_val': {'n_folds': n_folds},
        'classifier': classifier

    }

    if target == 'drug':
        target = drug
    else:
        target = stress

    if classifier == 'svm':
        clf = SVC()
        param_grid = svm_param_grid
    elif classifier == 'en':
        clf = SGDClassifier(loss='log', penalty='elasticnet', n_jobs=1)
        param_grid = en_param_grid

    timer = Timer()
    print('Starting...')
    pprint(log)

    split = StratifiedShuffleSplit(target['str'], n_iter=n_iter, test_size=test_size)
    grid = GridSearchCV(clf, param_grid=param_grid, cv=n_folds, n_jobs=1)
    accuracy = cross_val_score(grid, rma, y=target['str'], scoring='accuracy', cv=split,
                               n_jobs=n_iter, verbose=1)
    print('\n{}: Accuracy: {:.2%} +/- {:.2%}'.format(timer.elapsed(), np.nanmean(accuracy),
                                                     np.nanstd(accuracy)))

    log['results'] = {'accuracy': {'mean': accuracy.mean(), 'std': accuracy.std()},
                      'time': timer.elapsed()}
    log_results(log)
