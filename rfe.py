from __future__ import division, print_function
import argparse
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from data2 import load


class GridWithCoef(GridSearchCV):
    def fit(self, X, y=None):
        super(GridWithCoef, self).fit(X, y)
        self.coef_ = self.best_estimator_.coef_
        return self

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-path', default='./bucket/results/')
    parser.add_argument('--data')
    parser.add_argument('--target')
    parser.add_argument('--data-path', default='./bucket/data/')
    args = parser.parse_args()

    data, factors = load(args.data, data_path=args.data_path)

    clf = SVC(kernel='linear')
    
