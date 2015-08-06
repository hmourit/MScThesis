from __future__ import division, print_function

from data import load_mdd_data

if __name__ == '__main__':
    data, factors = load_mdd_data()

    from sklearn.preprocessing import LabelEncoder

    y = factors['stress']
    y = LabelEncoder().fit_transform(y)

    from feature_selection import mrmr_pool
    from preprocessing.discretization import ExpressionDiscretizer

    discr = ExpressionDiscretizer().fit(data).transform(data)

    feats = mrmr_pool(discr, y, select=10, pool_size=100, bins=3, verbose=True)
