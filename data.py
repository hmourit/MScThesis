from __future__ import division, print_function
import os
import cPickle as pickle
import pandas as pd


def _load_and_pickle_mdd_data(
        data_path='./bucket/data/mdd/',
        pickled_folder='pickled/',
        mdd_rma_pickled_file='rma.pickle',
        mdd_target_pickled_file='mdd_target.pickle'):
    fac = pd.read_csv(data_path + 'fac.csv', sep=';', index_col=0)
    rma = pd.read_csv(data_path + 'rma134.csv', index_col=0).transpose()
    numbers134 = rma.index.to_series().str.split('_').apply(lambda x: x[1]).astype(int)
    with open(os.path.join(data_path, pickled_folder, mdd_rma_pickled_file), 'wb') as out:
        pickle.dump(rma, out)

    # target = pd.DataFrame()
    # drug = fac[fac.number.isin(numbers134)]['drug']
    # rma['drug'] = drug.tolist()
    # target['drug'] = rma.pop('drug')
    # stress = fac[fac.number.isin(numbers134)]['stress']
    # rma['stress'] = stress.tolist()
    # target['stress'] = rma.pop('stress')
    fac = fac[fac.number.isin(numbers134)]
    fac['ProbeID'] = rma.index
    fac.set_index('ProbeID', inplace=True)
    with open(os.path.join(data_path, pickled_folder, mdd_target_pickled_file), 'wb') as out:
        pickle.dump(fac, out)

    return rma, fac


def load_mdd_data(
        data_path='./bucket/data/mdd/',
        pickled_folder='pickled/',
        mdd_rma_pickled_file='rma.pickle',
        mdd_target_pickled_file='mdd_target.pickle',
        log=None,
        verbose=False):
    rma_pickled_full_path = os.path.join(data_path, pickled_folder, mdd_rma_pickled_file)
    target_pickled_full_path = os.path.join(data_path, pickled_folder, mdd_target_pickled_file)
    if not os.path.isfile(rma_pickled_full_path) \
            or not os.path.isfile(target_pickled_full_path):
        if verbose:
            print('Pickled data not found, loading and pickling...')
        data, target = _load_and_pickle_mdd_data(data_path,
                                                 pickled_folder,
                                                 mdd_rma_pickled_file,
                                                 mdd_target_pickled_file)
    else:
        with open(rma_pickled_full_path, 'rb') as in_:
            data = pickle.load(in_)
        with open(target_pickled_full_path, 'rb') as in_:
            target = pickle.load(in_)

    if log is not None:
        log['data'] = 'mdd'

    return data, target


def _load_and_pickle_epi_ad_data(
        data_path='./bucket/data/epi_ad',
        pickled_folder='pickled/',
        betas_pickled_file='betas.pickle',
        factors_pickled_file='epi_ad_factors.pickle',
        betas_file='./GSE59685_betas.csv',
        factors_file='./GSE59685_series_matrix.txt'):
    betas_file = os.path.join(data_path, betas_file)
    betas = pd.read_table(betas_file, sep=',', header=[3, 4, 5], index_col=0,
                          memory_map=True, low_memory=True)
    betas = betas.transpose()
    betas.index = betas.index.get_level_values(1)

    with open(os.path.join(data_path, pickled_folder, betas_pickled_file), 'wb') as out:
        pickle.dump(betas, out)

    factors_file = os.path.join(data_path, factors_file)
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
    # factors.loc[factors['braak.stage'] == 'Exclude', 'braak.stage'] = np.nan
    # factors['braak.stage'] = factors['braak.stage'].astype(np.float_)

    with open(os.path.join(data_path, pickled_folder, factors_pickled_file), 'wb') as out:
        pickle.dump(factors, out)

    return betas, factors


def load_epi_ad_data(
        data_path='./bucket/data/epi_ad',
        pickled_folder='pickled/',
        betas_pickled_file='betas.pickle',
        factors_pickled_file='epi_ad_factors.pickle',
        betas_file='./GSE59685_betas.csv',
        factors_file='./GSE59685_series_matrix.txt',
        log=None,
        verbose=False,
        override_pickle=False):
    betas_pickled_full_path = os.path.join(data_path, pickled_folder, betas_pickled_file)
    factors_pickled_full_path = os.path.join(data_path, pickled_folder, factors_pickled_file)
    if (override_pickle or
            not os.path.isfile(betas_pickled_full_path) or
            not os.path.isfile(factors_pickled_full_path)):
        if verbose:
            print('Pickled data not found, loading and pickling...')
        data, factors = _load_and_pickle_epi_ad_data(data_path,
                                                     pickled_folder,
                                                     betas_pickled_file,
                                                     factors_pickled_file,
                                                     betas_file,
                                                     factors_file)
    else:
        with open(betas_pickled_full_path, 'rb') as in_:
            data = pickle.load(in_)
        with open(factors_pickled_full_path, 'rb') as in_:
            factors = pickle.load(in_)

    if log is not None:
        log['data'] = 'epi_ad'

    return data, factors
