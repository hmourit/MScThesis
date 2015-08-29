from __future__ import division, print_function
from datetime import datetime
import json
from socket import gethostname
import argparse
import re
import sys
import numpy as np
from os.path import join
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from data2 import load
from feature_selection import mrmr_iter
from preprocessing.discretization import ExpressionDiscretizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-path', default='./bucket/results/')
    parser.add_argument('--data')
    parser.add_argument('--target')
    parser.add_argument('--tissue',
                        type=lambda x: re.sub(r'[\"\']', '', x) if x is not None else None)
    parser.add_argument('--data-path', default='./bucket/data/')
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--test-size', type=float, default=0.1)
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
        sys.stdout.flush()

    load_params = {}
    if args.data == 'epi_ad':
        load_params = {'read_original': True, 'skip_pickle': True}

    data, factors = load(args.data, data_path=args.data_path, log=result, **load_params)
    if args.tissue:
        data = data[factors['source tissue'] == args.tissue]
        factors = factors[factors['source tissue'] == args.tissue]
    target = factors[args.target]
    target_num = LabelEncoder().fit_transform(target)

    split = StratifiedShuffleSplit(target, n_iter=1, test_size=args.test_size)
    train, test = next(split)

    data = data.iloc[train, :]
    target = target.iloc[train]

    result['test_samples'] = test.tolist()

    data = ExpressionDiscretizer().fit(data).transform(data)

    experiment_id = hash(json.dumps(result) + str(np.random.rand(10, 1)))
    result_file = join(args.results_path, 'subsets_mrmr_{}.json'.format(experiment_id))
    if args.verbose:
        print('Results will be saved to {}'.format(result_file))
        sys.stdout.flush()

    result['subsets'] = []

    features = []
    for i, f in enumerate(mrmr_iter(data, target_num, select=data.shape[1], bins=3)):
        features.append(f)
        if i >= 10 and i % 10 ** int(np.log10(i)) == 0:
            result['subsets'].append({
                'n_features': i,
                'features': features
            })
            with open(result_file, 'w') as f:
                json.dump(result, f, sort_keys=True, indent=2, separators=(',', ': '))
            print('{} features selected.')
            sys.stdout.flush()

    print('# OK')
