from __future__ import division, print_function
import os
import cPickle as pickle
import pandas as pd

data_path = './bucket/data/'
pickled_path = data_path + 'pickled/'
mdd_rma_pickled_file = 'rma.pickle'
mdd_target_pickled_file = 'mdd_target.pickle'


def load_and_pickle_mdd_data():

    fac = pd.read_csv(data_path + 'fac.csv', sep=';', index_col=0)
    rma = pd.read_csv(data_path + 'rma134.csv', index_col=0).transpose()
    numbers134 = rma.index.to_series().str.split('_').apply(lambda x: x[1]).astype(int)
    with open(pickled_path + mdd_rma_pickled_file, 'wb') as out:
        pickle.dump(rma, out)

    target = pd.DataFrame()
    drug = fac[fac.number.isin(numbers134)]['drug']
    rma['drug'] = drug.tolist()
    target['drug'] = rma.pop('drug')
    stress = fac[fac.number.isin(numbers134)]['stress']
    rma['stress'] = stress.tolist()
    target['stress'] = rma.pop('stress')
    with open(pickled_path + mdd_target_pickled_file, 'wb') as out:
        pickle.dump(target, out)

    return rma, target


def load_mdd_data():
    if not os.path.isfile(pickled_path + mdd_rma_pickled_file) \
       or not os.path.isfile(pickled_path + mdd_target_pickled_file):
        print('Pickled data not found, loading and pickling...')
        data, target = load_and_pickle_mdd_data()
    else:
        with open(pickled_path + mdd_rma_pickled_file, 'rb') as in_:
            data = pickle.load(in_)
        with open(pickled_path + mdd_target_pickled_file, 'rb') as in_:
            target = pickle.load(in_)
    return data, target

