from __future__ import division, print_function
import argparse
from datetime import datetime
import json
from os.path import join
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection.univariate_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from rfe2 import n_folds_parser, subset_sizes
from scripts.base import choose_classifier, GridWithCoef

from data2 import load


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
    result_file = join(args.results_path, 'fs_{}.json'.format(experiment_id))
    if args.verbose:
        print('Results will be saved to {}'.format(result_file))

    data, factors = load(args.data, data_path=args.data_path, log=result)
    target = factors[args.target]

    clf, param_grid = choose_classifier(args.clf, result, args.verbose)

    split = StratifiedShuffleSplit(target, n_iter=args.n_iter, test_size=args.test_size)
    n_features = data.shape[1]
    n_features_to_select = 10

    preprocess_steps = [('scaler', StandardScaler())]

    result['experiments'] = []
    for i, (train, test) in enumerate(split):
        result['experiments'].append({
            'iteration': i,
            'train_samples': data.index[train].tolist(),
            'subsets': []
        })
        scores_ = anova(data.iloc[train, :], target.iloc[train])
        for threshold in subset_sizes(n_features, n_features_to_select):
            features = np.argsort(scores_)[n_features - threshold:]
            pipeline = preprocess_steps + [('grid', GridWithCoef(clf, param_grid, cv=args.n_folds))]
            pipeline = Pipeline(pipeline)

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

        # Store results
        with open(result_file, 'w') as f:
            json.dump(result, f, sort_keys=True, indent=2, separators=(',', ': '))


def anova(train_data, train_target):
    selector = SelectKBest(k='all')
    selector.fit(train_data, train_target)
    scores_ = selector.scores_
    return scores_


if __name__ == '__main__':
    main()
