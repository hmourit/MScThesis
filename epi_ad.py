from __future__ import division, print_function
import argparse
import re
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import sys
import numpy as np
from sklearn.svm import SVC

from data import load_mdd_data, load_epi_ad_data
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from results import save_experiment
from utils_2 import Timer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_path')
    parser.add_argument('--data', default='epi_ad')
    parser.add_argument('--target', default='ad.disease.status')
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--test-size', type=float, default=0.1)
    parser.add_argument('--tissue', type=lambda x: re.sub(r'[\"\']', '', x))
    parser.add_argument('--clf', default='logit')
    parser.add_argument('--n-folds', type=int, default=10)
    parser.add_argument('--verbose', '-v', action='count')
    args = parser.parse_args()

    result = {}
    result.update(args.__dict__)
    if args.verbose:
        print(result)

    if args.data == 'epi_ad':
        betas, factors = load_epi_ad_data(log=result, verbose=args.verbose)
    elif args.data == 'mdd':
        betas, factors = load_mdd_data(log=result, verbose=args.verbose)
    else:
        result['error'] = '{} not a valid dataset.'.format(args.tissue)
        if args.verbose:
            print('{} not a valid dataset.'.format(args.tissue))

    if args.tissue is not None:
        condition = factors['source tissue'] == args.tissue
        if condition.sum() == 0:
            result['error'] = '{} is not a valid tissue.'.format(args.tissue)
            save_experiment(result, folder=args.results_path, filename=None, error=True,
                            verbose=args.verbose)
            sys.exit()
        betas = betas[condition]
        factors = factors[condition]

    target = factors[args.target]

    split = StratifiedShuffleSplit(target, n_iter=args.n_iter, test_size=args.test_size)
    result['split'] = {
        'type': 'StratifiedShuffleSplit',
        'n_iter': args.n_iter,
        'test_size': args.test_size
    }

    result['cross_val'] = {'n_folds': args.n_folds}

    pipeline = []

    if args.clf == 'logit':
        pass
    elif args.clf == 'en':
        alpha_range = np.logspace(-2, 7, 10)
        l1_ratio_range = np.arange(0., 1., 10)
        param_grid = dict(alpha=alpha_range, l1_ratio=l1_ratio_range)
        clf = SGDClassifier(loss='log', penalty='elasticnet')
        grid = GridSearchCV(clf, param_grid=param_grid, cv=args.n_folds, n_jobs=-1)
        pipeline.append(('en', grid))
    else:
        result['error'] = '{} is not a valid classifier.'.format(args.clf)
        save_experiment(result, folder=args.results_path, error=True,
                        verbose=args.verbose)

    pipeline = Pipeline(pipeline)

    timer = Timer()
    accuracy = cross_val_score(pipeline, betas, y=target, scoring='accuracy', cv=split, n_jobs=1)
    result['time'] = timer.elapsed()

    result['results'] = {
        'accuracy': {
            'scores': accuracy.tolist(),
            'mean': accuracy.mean(),
            'std': accuracy.std()
        }
    }

    save_experiment(result, folder=args.results_path, verbose=args.verbose)
