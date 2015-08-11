from __future__ import division, print_function
import argparse
from datetime import datetime
from time import time
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import RFE

from data2 import load
from scripts.base import choose_classifier, GridWithCoef

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-path', default='./bucket/results/')
    parser.add_argument('--data')
    parser.add_argument('--target')
    parser.add_argument('--data-path', default='./bucket/data/')
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--clf')
    args = parser.parse_args()

    result = {}
    result.update(args.__dict__)
    if args.verbose:
        for key in result:
            print('# {}: {}'.format(key, result[key]))
        print('# Start: ' + datetime.now().strftime("%d/%m/%y %H:%M:%S"))

    data, factors = load(args.data, data_path=args.data_path)
    target = factors[args.target]

    clf, param_grid = choose_classifier(args.clf, log=result)

    grid = GridWithCoef(clf, param_grid)

    split = StratifiedShuffleSplit(target, n_iter=args.n_iter, test_size=args.test_size)
    result['split'] = {
        'type': 'StratifiedShuffleSplit',
        'n_iter': args.n_iter,
        'test_size': args.test_size
    }

    # FIXME
    n_features_to_select = 10000
    step = 10000
    ###

    t0 = time()
    for train, test in split:
        rfe = RFE(grid, n_features_to_select=n_features_to_select, step=step)
        rfe.fit(data.iloc[:, train], target.iloc[:, train])




