from __future__ import division, print_function
import argparse
from datetime import datetime
import json
from time import time
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.cross_validation import StratifiedShuffleSplit, LeaveOneOut
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.utils import safe_sqr
from data2 import load
from results import save_experiment2
from scripts.base import choose_classifier, GridWithCoef
from os.path import join
import sys


class FooClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        self.coef_ = np.random.rand(X.shape[1])


def n_folds_parser(x):
    if x.lower() == 'loo':
        return 'loo'
    else:
        return int(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-path', default='./bucket/results/')
    parser.add_argument('--data')
    parser.add_argument('--target')
    parser.add_argument('--data-path', default='./bucket/data/')
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--test-size', type=float, default=0.1)
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--n-folds', default=10, type=n_folds_parser)
    parser.add_argument('--clf')
    args = parser.parse_args()

    result = {}
    result.update(args.__dict__)
    if args.verbose:
        for key in result:
            print('# {}: {}'.format(key, result[key]))
        print('# Start: ' + datetime.now().strftime("%d/%m/%y %H:%M:%S"))
    result['selections'] = []

    experiment_id = hash(json.dumps(result) + str(np.random.rand(10)))
    result_file = join(args.results_path, 'fs_{}.json'.format(experiment_id))
    if args.verbose:
        print('Results will be saved to {}'.format(result_file))

    data, factors = load(args.data, data_path=args.data_path, log=result)
    target = factors[args.target]

    clf, param_grid = choose_classifier(args.clf, result, args.verbose)

    feature_names = data.columns

    split = StratifiedShuffleSplit(target, n_iter=args.n_iter, test_size=args.test_size)
    n_features = data.shape[1]
    n_features_to_select = 1

    support_ = np.ones(n_features, dtype=np.bool)
    ranking_ = np.ones(n_features, dtype=np.int)
    # Elimination
    t0 = time()
    d0 = datetime.now()
    while np.sum(support_) > n_features_to_select:
        step = 10 ** int(np.log10(np.sum(support_) - 1))
        odd_step = np.sum(support_) - step * (np.sum(support_) // step)
        if odd_step > 0:
            step = odd_step

        if args.verbose:
            print('[{}] Selecting best {:d} features.'
                  .format(datetime.now() - d0, np.sum(support_) - step))
        # Remaining features
        features = np.arange(n_features)[support_]

        coef_ = None
        test_scores = []
        for train, test in split:
            # Rank the remaining features
            if args.n_folds == 'loo':
                cv = LeaveOneOut(len(train))
            else:
                cv = args.n_folds
            estimator = GridWithCoef(clf, param_grid, cv=cv)

            estimator.fit(data.iloc[train, features], target.iloc[train])
            if coef_ is None:
                coef_ = safe_sqr(estimator.coef_)
            else:
                coef_ += safe_sqr(estimator.coef_)

            test_scores.append(estimator.score(data.iloc[test, features], target.iloc[test]))

        if coef_.ndim > 1:
            ranks = np.argsort(coef_.sum(axis=0))
        else:
            ranks = np.argsort(coef_)

        # for sparse case ranks is matrix
        ranks = np.ravel(ranks)

        # Eliminate the worse features
        threshold = min(step, np.sum(support_) - n_features_to_select)
        support_[features[ranks][:threshold]] = False
        ranking_[np.logical_not(support_)] += 1

        result['selections'].append({
            'scores': test_scores,
            'n_features': np.sum(support_),
            'features': feature_names[support_].tolist()
        })

        with open(result_file, 'w') as f:
            json.dump(result, f, sort_keys=True, indent=2, separators=(',', ': '))

    if args.verbose:
        print('# OK')


if __name__ == '__main__':
    main()
