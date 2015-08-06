from __future__ import division, print_function


class ExpressionDiscretizer(object):
    def fit(self, X, y=None):
        self.mean_ = X.values.mean(axis=0)
        self.std_ = X.values.std(axis=0)
        return self

    def transform(self, X):
        over = X > self.mean_ + self.std_ / 2
        under = X < self.mean_ - self.std_ / 2
        X[over] = 0
        X[under] = 1
        X[~over & ~under] = 2
        return X
