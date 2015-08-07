from __future__ import division, print_function
from os.path import join, isfile
import cPickle as pickle
import pandas as pd

DATA_PATH = './bucket/data/'
PICKLED_FOLDER = 'pickled'
PICKLED_DATA_FILE = 'data.pickle'
PICKLED_FACTORS_FILE = 'factors.pickle'

MDD_DATA_FILE = 'rma134.csv'
MDD_FACTORS_FILE = 'fac.csv'


def _load_mdd(data_path=DATA_PATH):
    data_file = join(data_path, 'mdd', MDD_DATA_FILE)
    factors_file = join(data_path, 'mdd', MDD_FACTORS_FILE)

    factors = pd.read_csv(factors_file, sep=';', index_col=0)
    data = pd.read_csv(data_file, index_col=0).transpose()
    numbers134 = data.index.to_series().str.split('_').apply(lambda x: x[1]).astype(int)

    factors = factors[factors.number.isin(numbers134)]
    factors['SampleID'] = data.index
    factors.set_index('SampleID', inplace=True)
    return data, factors


DISCARDED_SAMPLES = [145, 146, 147, 148, 149, 150, 152, 153, 155, 156]
MDD_RAW_DATA_FILE = 'data.csv'
MDD_RAW_FACTORS_FILE = 'nfac.csv'


def _load_mdd_raw(data_path=DATA_PATH):
    data_file = join(data_path, 'mdd_raw', MDD_RAW_DATA_FILE)
    factors_file = join(data_path, 'mdd_raw', MDD_RAW_FACTORS_FILE)

    data = pd.read_csv(data_file, index_col=0).transpose()
    factors = pd.read_csv(factors_file, sep=',', index_col=0)
    factors['SampleID'] = data.index
    factors.set_index('SampleID', inplace=True)
    factors = factors[~factors.number.isin(DISCARDED_SAMPLES)]

    data = data.loc[factors.index]

    return data, factors


EPI_AD_DATA_FILE = 'GSE59685_betas.csv'
EPI_AD_FACTORS_FILE = 'GSE59685_series_matrix.txt'


def _load_epi_ad(data_path=DATA_PATH):
    data_file = join(data_path, 'epi_ad', EPI_AD_DATA_FILE)
    factors_file = join(data_path, 'epi_ad', EPI_AD_FACTORS_FILE)

    data = pd.read_table(data_file, sep=',', header=[3, 4, 5], index_col=0,
                         memory_map=True, low_memory=True)
    data = data.transpose()
    data.index = data.index.get_level_values(1)

    factors = pd.read_table(factors_file,
                            sep='\t', header=None, skiprows=49,
                            skip_footer=1, engine='python')

    factors.ix[:, 0] = factors.ix[:, 0].apply(lambda x: x.replace('!Sample_', ''))
    factors.columns = factors.iloc[-1].tolist()
    factors.set_index('ID_REF', inplace=True)
    factors.drop('ID_REF', inplace=True)
    factors = factors.transpose()
    tmp = factors['characteristics_ch1']
    factors.drop('characteristics_ch1', axis=1, inplace=True)
    for i in range(len(tmp.columns)):
        name = tmp.iloc[0, i].split(':')[0]
        factors[name] = tmp.iloc[:, i].apply(lambda x: x.split(':')[1].strip())
    factors['subjectid'] = factors['subjectid'].astype(int)

    return data, factors


def load(dataset,
         data_path=DATA_PATH,
         override_pickle=False,
         log=None):
    path = join(data_path, dataset)
    pickled_data = join(path, PICKLED_FOLDER, PICKLED_DATA_FILE)
    pickled_factors = join(path, PICKLED_FOLDER, PICKLED_FACTORS_FILE)

    if (override_pickle or
            not isfile(pickled_data) or
            not isfile(pickled_factors)):
        print('Loading and pickling')
        if dataset == 'mdd':
            data, factors = _load_mdd(data_path)
        elif dataset == 'mdd_raw':
            data, factors = _load_mdd_raw(data_path)
        elif dataset == 'epi_ad':
            data, factors = _load_epi_ad(data_path)
        else:
            raise ValueError('Dataset {} unknown.'.format(dataset))
        _pickle_files(dataset, data, factors, data_path=data_path)
    else:
        with open(pickled_data, 'rb') as f:
            data = pickle.load(f)
        with open(pickled_factors, 'rb') as f:
            factors = pickle.load(f)

    if log is not None:
        log['data'] = dataset

    return data, factors


def _pickle_files(dataset,
                  data,
                  factors,
                  data_path=DATA_PATH):
    path = join(data_path, dataset)
    pickled_data = join(path, PICKLED_FOLDER, PICKLED_DATA_FILE)
    pickled_factors = join(path, PICKLED_FOLDER, PICKLED_FACTORS_FILE)

    with open(pickled_data, 'wb') as f:
        pickle.dump(data, f)

    with open(pickled_factors, 'wb') as f:
        pickle.dump(factors, f)
