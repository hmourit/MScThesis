from __future__ import division, print_function
import argparse
from time import time
from datetime import datetime

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit, LeaveOneOut

from data2 import load
from results import save_experiment2
from scripts.base import choose_classifier


def n_folds_parser(x):
    if x.lower() == 'loo':
        return 'loo'
    else:
        return int(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-path', default='./bucket/results/')
    parser.add_argument('--data')
    parser.add_argument('--target')
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--test-size', type=float, default=0.1)
    # parser.add_argument('--tissue', type=lambda x: re.sub(r'[\"\']', '', x))
    parser.add_argument('--clf')
    parser.add_argument('--n-folds', default=10, type=n_folds_parser)
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--data-path', default='./bucket/data/')
    args = parser.parse_args()

    result = {}
    result.update(args.__dict__)
    if args.verbose:
        for key in result:
            print('# {}: {}'.format(key, result[key]))
        print('# Start: ' + datetime.now().strftime("%d/%m/%y %H:%M:%S"))

    data, factors = load(args.data, data_path=args.data_path, log=result)
    target = factors[args.target]

    # if args.tissue is not None:
    #     condition = factors['source tissue'] == args.tissue
    #     if condition.sum() == 0:
    #         result['error'] = '{} is not a valid tissue.'.format(args.tissue)
    #         save_experiment(result, folder=args.results_path, filename=None, error=True,
    #                         verbose=args.verbose)
    #         sys.exit()
    #     data = data[condition]
    #     factors = factors[condition]

    split = StratifiedShuffleSplit(target, n_iter=args.n_iter, test_size=args.test_size)
    result['split'] = {
        'type': 'StratifiedShuffleSplit',
        'n_iter': args.n_iter,
        'test_size': args.test_size
    }

    result['cross_val'] = {'n_folds': args.n_folds}

    steps = []

    clf, param_grid = choose_classifier(args.clf)

    result['results'] = {
        'accuracy': {
            'train': [],
            'test': []
        }
    }

    steps.append('clf')

    if args.verbose:
        print('{:<7} {:<7} {}'.format('Train', 'Test', 'Time'))
    t0 = time()
    for train, test in split:
        if args.n_folds == 'loo':
            cv = LeaveOneOut(len(train))
        else:
            cv = args.n_folds
        grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, n_jobs=-1)
        steps[-1] = ('clf', grid)
        pipeline = Pipeline(steps)

        pipeline.fit(data.iloc[train, :], y=target.iloc[train])
        train_score = pipeline.score(data.iloc[train, :], y=target.iloc[train])
        test_score = pipeline.score(data.iloc[test, :], y=target.iloc[test])
        result['results']['accuracy']['train'].append(train_score)
        result['results']['accuracy']['test'].append(test_score)
        if args.verbose:
            print('{:0.5f} {:0.5f} {}'.format(train_score, test_score, time() - t0))

    if args.verbose:
        scores = result['results']['accuracy']['train']
        print('# Train score: {} +/- {}'.format(np.mean(scores), np.std(scores)))
        scores = result['results']['accuracy']['test']
        print('# Test score: {} +/- {}'.format(np.mean(scores), np.std(scores)))
        print('# Total time: {}'.format(time() - t0))

    save_experiment2(result, folder=args.results_path, verbose=args.verbose)

    if args.verbose:
        print("# OK")
