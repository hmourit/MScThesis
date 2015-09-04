from __future__ import division, print_function
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC


def choose_classifier(estimator, log=None, verbose=False):
    if estimator == 'en':
        alpha_range = np.logspace(-2, 7, 10)
        l1_ratio_range = np.arange(0., 1., 10)
        param_grid = dict(alpha=alpha_range, l1_ratio=l1_ratio_range)
        clf = SGDClassifier(loss='log', penalty='elasticnet')
    elif estimator in {'svm', 'svm_rbf'}:
        c_range = np.logspace(-2, 7, 10)
        gamma_range = np.logspace(-6, 3, 10)
        param_grid = dict(gamma=gamma_range, C=c_range)
        clf = SVC(cache_size=1000)
    elif estimator == 'svm_linear_kernel':
        c_range = np.logspace(-2, 7, 10)
        param_grid = dict(C=c_range)
        clf = SVC(kernel='linear')
    elif estimator == 'svm_linear':
        c_range = np.logspace(-2, 7, 10)
        param_grid = dict(C=c_range)
        clf = LinearSVC(penalty='l2')
    elif estimator == 'svm_linear_l1':
        c_range = np.logspace(-2, 7, 10)
        param_grid = dict(C=c_range)
        clf = LinearSVC(penalty='l1', dual=False)
    else:
        # print('# ERROR: {} is not a valid classifier.'.format(estimator))
        # sys.exit(1)
        raise ValueError('{} is not a valid classifier.'.format(estimator))

    return clf, param_grid


class GridWithCoef(GridSearchCV):
    def fit(self, X, y=None):
        super(GridWithCoef, self).fit(X, y)
        self.coef_ = self.best_estimator_.coef_
        return self