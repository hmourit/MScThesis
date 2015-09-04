from __future__ import division, print_function
import json
from socket import gethostname
import argparse
from datetime import datetime
import sys
from os.path import join

import numpy as np

from data2 import load
from scripts.base import choose_classifier, GridWithCoef


def n_folds_parser(x):
    if x.lower() == 'loo':
        return 'loo'
    else:
        return int(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-path', default='./bucket/results/')
    parser.add_argument('--data-path', default='./bucket/data/')
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--test-size', type=float, default=0.1)
    parser.add_argument('--n-folds', default=10, type=n_folds_parser)
    parser.add_argument('--clf')
    parser.add_argument('--mrmr-result')
    args = parser.parse_args()

    result = {}
    result.update(args.__dict__)
    start_time = datetime.now().strftime("%d/%m/%y %H:%M:%S")
    result['start_time'] = start_time
    try:
        mrmr_results = json.load(open(args.mrmr_result, 'r'))
    except:
        print("mRMR result file couldn't be loaded.")
        sys.stdout.flush()
        sys.exit()
    result['data'] = mrmr_results['data']
    result['tissue'] = mrmr_results['tissue']
    result['target'] = mrmr_results['target']
    result['test_samples'] = mrmr_results['test_samples']
    if args.verbose:
        for key in result:
            print('# {}: {}'.format(key, result[key]))
        print('# Running in: ' + gethostname())
        print('# Start: ' + start_time)
        sys.stdout.flush()

    load_params = {}
    if result['data'] == 'epi_ad':
        load_params = {'read_original': True, 'skip_pickle': True}

    data, factors = load(result['data'], data_path=args.data_path, log=result, **load_params)
    if result['tissue']:
        data = data[factors['source tissue'] == result['tissue']]
        factors = factors[factors['source tissue'] == result['tissue']]
    target = factors[result['target']]

    clf, param_grid = choose_classifier(args.clf, result, args.verbose)

    train_samples = np.ones(data.shape[0], dtype=np.bool)
    train_samples[result['test_samples']] = False

    train_data = data.loc[train_samples, :]
    train_target = target.loc[train_samples]
    test_data = data.iloc[result['test_samples'], :]
    test_target = target.iloc[result['test_samples']]

    experiment_id = hash(json.dumps(result) + str(np.random.rand(10, 1)))
    result_file = join(args.results_path, 'mrmr_{}.json'.format(experiment_id))
    if args.verbose:
        print('Results will be saved to {}'.format(result_file))
        sys.stdout.flush()

    max_features = mrmr_results['subsets'][-1]['n_features']
    all_features = mrmr_results['subsets'][-1]['features']

    result['experiments'] = [{
        'iteration': 0,
        'subsets': []
    }]
    d0 = datetime.now()
    current_size = 10
    while current_size <= max_features:
        if args.verbose:
            print('[{}] Fitting with {} features.'.format(datetime.now() - d0, current_size))
            sys.stdout.flush()

        features = all_features[:current_size]

        grid = GridWithCoef(clf, param_grid, cv=args.n_folds)
        grid.fit(train_data.iloc[:, features], train_target)

        # Save results for current set of features
        result['experiments'][-1]['subsets'].append({
            'n_features': current_size,
            'features': data.columns[features].tolist(),
            'best_params': grid.best_params_,
            'train': {
                'y_true': train_target.tolist(),
                'y_pred': grid.predict(train_data.iloc[:, features]).tolist()
            },
            'test': {
                'y_true': test_target.tolist(),
                'y_pred': grid.predict(test_data.iloc[:, features]).tolist()
            }
        })

        # Store results
        with open(result_file, 'w') as f:
            json.dump(result, f, sort_keys=True, indent=2, separators=(',', ': '))

        current_size += 10 ** int(np.log10(current_size))

    if args.verbose:
        print('# OK')
        sys.stdout.flush()