from __future__ import division, print_function
import argparse

from data2 import load
from sklearn.preprocessing import LabelEncoder
from feature_selection import mrmr
from preprocessing.discretization import ExpressionDiscretizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--target')
    args = parser.parse_args()

    print('Data: {}'.format(args.data))
    print('Target: {}'.format(args.target))

    data, factors = load(args.data)

    y = factors[args.target]
    y = LabelEncoder().fit_transform(y)

    discr = ExpressionDiscretizer().fit(data).transform(data)

    feats = mrmr(discr, y, select=data.shape[1], bins=3, verbose=True)
