from __future__ import division, print_function
from socket import gethostname
import argparse
from datetime import datetime
import json
from os.path import join
import numpy as np
import re
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.label import LabelEncoder
from feature_selection import relevance
from preprocessing.discretization import ExpressionDiscretizer
from rfe2 import n_folds_parser, subset_sizes
from scripts.base import choose_classifier, GridWithCoef

from data2 import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-path', default='./bucket/results/')
    parser.add_argument('--data')
    parser.add_argument('--target')
    parser.add_argument('--tissue',
                        type=lambda x: re.sub(r'[\"\']', '', x) if x is not None else None)
    parser.add_argument('--data-path', default='./bucket/data/')
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--test-size', type=float, default=0.1)
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--n-folds', default=10, type=n_folds_parser)
    parser.add_argument('--clf')
    parser.add_argument('--filter', default='anova')
    args = parser.parse_args()

    result = {}
    result.update(args.__dict__)
    start_time = datetime.now().strftime("%d/%m/%y %H:%M:%S")
    result['start_time'] = start_time
    if args.verbose:
        for key in result:
            print('# {}: {}'.format(key, result[key]))
        print('# Running in: ' + gethostname())
        print('# Start: ' + start_time)

    load_params = {}
    if args.data == 'epi_ad':
        load_params = {'read_original': True, 'skip_pickle': True}

    data, factors = load(args.data, data_path=args.data_path, log=result, **load_params)
    if args.tissue:
        data = data[factors['source tissue'] == args.tissue]
        factors = factors[factors['source tissue'] == args.tissue]
    target = factors[args.target]
    target_num = LabelEncoder().fit_transform(target)

    clf, param_grid = choose_classifier(args.clf, result, args.verbose)

    score_params = {}
    preprocessor = None
    if args.filter == 'anova':
        score_features = anova
    elif args.filter == 'infogain_10':
        score_features = relevance
        score_params = {'bins': 10}
    elif args.filter == 'infogain_exp':
        preprocessor = ExpressionDiscretizer()
        score_features = relevance
        score_params = {'bins': 3}
    elif args.filter == 'chi2':
        preprocessor = ExpressionDiscretizer()
        score_features = chi_squared
    else:
        raise ValueError('Filter {} unknown.'.format(args.filter))

    experiment_id = hash(json.dumps(result) + str(np.random.rand(10, 1)))
    result_file = join(args.results_path, '{}_{}.json'.format(args.filter, experiment_id))
    if args.verbose:
        print('Results will be saved to {}'.format(result_file))

    split = StratifiedShuffleSplit(target, n_iter=args.n_iter, test_size=args.test_size)
    n_features = data.shape[1]
    n_features_to_select = 9

    d0 = datetime.now()
    result['experiments'] = []
    for i, (train, test) in enumerate(split):
        if args.verbose:
            print('### ITERATION {}'.format(i))
        if preprocessor:
            preprocessor.fit(data.iloc[train, :])
            train_data = preprocessor.transform(data.iloc[train, :])
            test_data = preprocessor.transform(data.iloc[test, :])
        else:
            train_data = data.iloc[train, :]
            test_data = data.iloc[test, :]

        scores_ = score_features(train_data, target_num[train], **score_params)
        result['experiments'].append({
            'iteration': i,
            'train_samples_label': data.index[train].tolist(),
            'train_samples_idx': train.tolist(),
            'scores': scores_.tolist()
        })
        if args.verbose:
            print('[{}] Features scored.'.format(datetime.now() - d0))

        result['experiments'][-1]['subsets'] = []
        current_size = n_features
        sorted_features = np.argsort(scores_)
        for step in subset_sizes(n_features, n_features_to_select):
            if args.verbose:
                print('[{}] Fitting with {} features.'.format(datetime.now() - d0, current_size))

            features = sorted_features[-current_size:]

            grid = GridWithCoef(clf, param_grid, cv=args.n_folds)
            grid.fit(train_data.iloc[:, features], target.iloc[train])

            # Save results for current set of features
            result['experiments'][-1]['subsets'].append({
                'n_features': current_size,
                'features': data.columns[features].tolist(),
                'best_params': grid.best_params_,
                'train': {
                    'y_true': target.iloc[train].tolist(),
                    'y_pred': grid.predict(train_data.iloc[:, features]).tolist()
                },
                'test': {
                    'y_true': target.iloc[test].tolist(),
                    'y_pred': grid.predict(test_data.iloc[:, features]).tolist()
                }
            })

            # Store results
            with open(result_file, 'w') as f:
                json.dump(result, f, sort_keys=True, indent=2, separators=(',', ': '))

            current_size -= step

    if args.verbose:
        print('# OK')


def anova(train_data, train_target):
    selector = SelectKBest(k='all')
    selector.fit(train_data, train_target)
    scores_ = selector.scores_
    return scores_


def chi_squared(train_data, train_target):
    selector = SelectKBest(k='all', score_func=chi2)
    selector.fit(train_data, train_target)
    scores_ = selector.scores_
    return scores_

if __name__ == '__main__':
    main()
