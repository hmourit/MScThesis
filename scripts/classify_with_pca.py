import sys
from pprint import pprint

import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score

from utils import Timer
from results import save_experiment
import data


def main(argv):
    # rma, drug, stress = load_data()
    rma, label = data.load_mdd_data()

    alpha_range = np.logspace(-2, 7, 10)
    l1_ratio_range = np.arange(0., 1., 0.1)
    en_param_grid = dict(alpha=alpha_range, l1_ratio=l1_ratio_range)

    c_range = np.logspace(-2, 7, 10)
    gamma_range = np.logspace(-6, 3, 10)
    svm_param_grid = dict(gamma=gamma_range, C=c_range)

    logit_param_grid = dict(C=c_range)

    test_size = float(argv[1])
    n_iter = int(argv[2])
    n_folds = int(argv[3])
    target = argv[4]
    classifier = argv[5]
    pca_components = int(argv[7])
    log = {
        'target': target,
        'pca': {'n_components': pca_components},
        'split': {
            'type': 'StratifiedShuffleSplit',
            'n_iter': n_iter,
            'test_size': test_size
        },
        'cross_val': {'n_folds': n_folds},
        'classifier': classifier

    }
    # if target == 'drug':
    #     target = drug
    # else:
    #     target = stress

    pca = PCA(n_components=pca_components)

    if classifier == 'svm':
        clf = SVC()
        param_grid = svm_param_grid
        grid_search = True
    elif classifier == 'en':
        clf = SGDClassifier(loss='log', penalty='elasticnet', n_jobs=1)
        param_grid = en_param_grid
        grid_search = True
    elif classifier == 'logit':
        clf = LogisticRegression()
        param_grid = logit_param_grid
        grid_search = True

    timer = Timer()
    print('\nStarting...' + ' '.join(argv))
    pprint(log)
    split = StratifiedShuffleSplit(label[target], n_iter=n_iter, test_size=test_size)

    if grid_search:
        clf = GridSearchCV(clf, param_grid=param_grid, cv=n_folds, n_jobs=1)

    pipeline = Pipeline([('pca', pca), ('clf', clf)])

    accuracy = cross_val_score(pipeline, rma, y=label[target], scoring='accuracy', cv=split,
                               n_jobs=n_iter, verbose=1)
    print('\n{}: Accuracy: {:.2%} +/- {:.2%}'.format(timer.elapsed(), np.nanmean(accuracy),
                                                     np.nanstd(accuracy)))

    log['results'] = {
        'accuracy': {
            'scores': accuracy.tolist(),
            'mean': accuracy.mean(),
            'std': accuracy.std()
        }
    }
    log['time'] = timer.elapsed()

    # results = [dict(log, accuracy=acc) for acc in accuracy]
    # log_results(results)
    # save_results(results, folder=argv[6], filename='results_new.json')
    save_experiment(log, folder=argv[6])

if __name__ == '__main__':
    main(sys.argv)
