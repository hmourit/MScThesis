from __future__ import division, print_function
import argparse
import json
from datetime import datetime
from os.path import join

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import safe_sqr

from data2 import load
from scripts.base import choose_classifier, GridWithCoef


class FooClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        self.coef_ = np.random.rand(X.shape[1])


def n_folds_parser(x):
    if x.lower() == 'loo':
        return 'loo'
    else:
        return int(x)


def subset_sizes(n_features, n_features_to_select):
    selected = n_features
    while selected > n_features_to_select:
        step = 10 ** int(np.log10(selected - 1))
        odd_step = selected - step * (selected // step)
        if odd_step > 0:
            step = odd_step
        selected -= step
        yield selected


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
    start_time = datetime.now().strftime("%d/%m/%y %H:%M:%S")
    result['start_time'] = start_time
    if args.verbose:
        for key in result:
            print('# {}: {}'.format(key, result[key]))
        print('# Start: ' + start_time)

    experiment_id = hash(json.dumps(result) + str(np.random.rand(10, 1)))
    result_file = join(args.results_path, 'rfe_{}.json'.format(experiment_id))
    if args.verbose:
        print('Results will be saved to {}'.format(result_file))

    data, factors = load(args.data, data_path=args.data_path, log=result)
    target = factors[args.target]

    clf, param_grid = choose_classifier(args.clf, result, args.verbose)

    split = StratifiedShuffleSplit(target, n_iter=args.n_iter, test_size=args.test_size)
    n_features = data.shape[1]
    n_features_to_select = 10

    preprocess_steps = [('scaler', StandardScaler())]

    # RFE
    result['experiments'] = []
    for i, (train, test) in enumerate(split):
        result['experiments'].append({
            'iteration': i,
            'train_samples': data.index[train].tolist(),
            'subsets': []
        })
        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)
        for threshold in subset_sizes(n_features, n_features_to_select):
            # Train with current subset
            pipeline = preprocess_steps + [('grid', GridWithCoef(clf, param_grid, cv=args.n_folds))]
            pipeline = Pipeline(pipeline)

            features = np.arange(n_features)[support_]
            pipeline.fit(data.iloc[train, features], target.iloc[train])

            # Save results for current set of features
            grid = pipeline.steps[-1][1]
            result['experiments'][-1]['subsets'].append({
                'features': data.columns[features].tolist(),
                'best_params': grid.best_params_,
                'train': {
                    'y_true': target.iloc[train].tolist(),
                    'y_pred': grid.predict(data.iloc[train, features]).tolist()
                },
                'test': {
                    'y_true': target.iloc[test].tolist(),
                    'y_pred': grid.predict(data.iloc[test, features]).tolist()
                }
            })

            # Select best subset
            coef_ = safe_sqr(grid.coef_)

            if coef_.ndim > 1:
                ranks = np.argsort(coef_.sum(axis=0))
            else:
                ranks = np.argsort(coef_)

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1

            # Store results
            with open(result_file, 'w') as f:
                json.dump(result, f, sort_keys=True, indent=2, separators=(',', ': '))

    if args.verbose:
        print('# OK')


if __name__ == '__main__':
    main()
