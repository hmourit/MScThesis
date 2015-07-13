
# coding: utf-8

# In[1]:

from __future__ import division, print_function


# In[2]:

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


# In[3]:

get_ipython().magic(u'matplotlib inline')
pd.set_option("max_rows", 10)
from pprint import pprint


# In[4]:

from sklearn.cross_validation import KFold, cross_val_score, cross_val_predict
from sklearn.grid_search import GridSearchCV


# In[5]:

from pandas_confusion import ConfusionMatrix


# In[6]:

SEED = 10
kf = KFold(n=134, n_folds=20, shuffle=True, random_state=SEED)


# ## Load data

# In[13]:

fac = pd.read_csv('./fac.csv', sep=';', index_col=0)
rma = pd.read_csv('./rma134.csv', index_col=0).transpose()
numbers134 = rma.index.to_series().str.split('_').apply(lambda x: x[1]).astype(int)
with open('rma.pickle', 'wb') as out:
    pickle.dump(rma, out)
rma.head()


# In[8]:

rma.describe()


# ### Targets

# In[14]:

DRUGS = ['N', 'E', 'C']
DRUG_MAP = {'N': 0, 'E': 1, 'C': 2}

drug = fac[fac.number.isin(numbers134)]['drug']
rma['drug'] = drug.tolist()
drug = pd.DataFrame()
drug['str'] = rma.pop('drug')
drug['int'] = drug['str'].replace(DRUG_MAP)
drug = drug.join(drug['str'].str.get_dummies())
with open('drug.pickle', 'wb') as out:
    pickle.dump(drug, out)
drug.head()


# In[26]:

stress = fac[fac.number.isin(numbers134)]['stress']
rma['stress'] = stress.tolist()
stress = pd.DataFrame()
stress['str'] = rma.pop('stress')
with open('stress.pickle', 'wb') as out:
    pickle.dump(stress, out)
stress.head()


# ## SVM

# In[10]:

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Find best parameters:

# In[11]:

retrain = False
if retrain:
    C_range = np.logspace(-2, 7, 10)
    gamma_range = np.logspace(-6, 3, 10)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=kf, n_jobs=-1)
    grid.fit(rma, drug['str'])

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    best_params = grid.best_params_
else:
    best_params = {'C': 1000, 'gamma': 1e-5}


# Show results:

# In[12]:

clf = SVC(**best_params)
pred = cross_val_predict(clf, rma, y=drug['str'], cv=kf, n_jobs=-1)


# In[13]:

print('Accuracy:', accuracy_score(drug['str'], pred), '\n')
print(classification_report(drug['str'], pred))
# print(confusion_matrix(drug['str'], pred, labels=DRUGS))
ConfusionMatrix(drug['str'].values, pred).plot()
plt.savefig('svm_conf_mat.eps')


# ### ElasticNet

# ### Implementation A

# ElasticNet logistic regression for multiclass classification

# In[14]:

from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler


# In[15]:

RETRAIN_ANYWAY = False
TRAINED_GRID_FILE = 'EN_grid.pickle'

clf = SGDClassifier(loss='log', penalty='elasticnet', n_jobs=-1)
if RETRAIN_ANYWAY or not os.path.exists(TRAINED_GRID_FILE):
    print('training')
    alpha_range = np.logspace(-2, 7, 10)
    l1_ratio_range = np.arange(0., 1., 0.1)
    param_grid = dict(alpha=alpha_range, l1_ratio=l1_ratio_range)
    grid = GridSearchCV(clf, param_grid=param_grid, cv=kf, n_jobs=-1)
    grid.fit(rma, drug['str'])
    
    with open(TRAINED_GRID_FILE, 'wb') as out:
        pickle.dump(grid, out)
else:
    print('read')
    with open(TRAINED_GRID_FILE, 'rb') as in_:
        grid = pickle.load(in_)

best_params = grid.best_params_

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


# Show results:

# In[16]:

clf = SGDClassifier(loss='log', penalty='elasticnet', n_jobs=-1, **best_params)
clf.fit(rma, drug['str'])
pred = cross_val_predict(clf, rma, y=drug['str'], cv=kf, n_jobs=-1)


# In[17]:

print('Accuracy:', accuracy_score(drug['str'], pred), '\n')
print(classification_report(drug['str'], pred))
# print(confusion_matrix(drug['str'], pred, labels=DRUGS))
ConfusionMatrix(drug['str'].values, pred).plot()
plt.savefig('en_conf_mat.eps')


# ### Implementation C

# In[18]:

from sklearn.feature_selection import RFECV, RFE


# In[19]:

clf = SGDClassifier(loss='log', penalty='elasticnet', n_jobs=-1, **best_params)
rfe = RFECV(clf, step=100, verbose=1)
rfe.fit(rma, drug['str'])


# In[21]:

rfe2 = RFE(clf, n_features_to_select=200, step=100)
rfe2.fit(rma, drug['str'])


# In[22]:

rfe3 = RFECV(clf, step=10)
rfe3.fit(rma.iloc[:, rfe2.support_], drug['str'])
print(rfe3.n_features_)


# In[27]:

support = np.zeros((rma.shape[1],), dtype=bool)
for n, (train_ix, test_ix) in enumerate(kf):
    print('Fold {}'.format(n))
    rfe_tmp = RFE(clf, n_features_to_select=80, step=100)
    rfe_tmp.fit(rma.iloc[train_ix], drug.ix[train_ix, 'str'])
    print('{} features selected'.format(rfe_tmp.n_features_))
    support |= rfe_tmp.support_


# In[59]:

support.shape


# In[30]:

clf = SGDClassifier(loss='log', penalty='elasticnet', n_jobs=-1, **best_params)
pred = cross_val_predict(clf, rma.iloc[:, support], y=drug['str'], cv=kf, n_jobs=-1)


# In[31]:

print('Accuracy:', accuracy_score(drug['str'], pred), '\n')
print(classification_report(drug['str'], pred))
# print(confusion_matrix(drug['str'], pred, labels=DRUGS))
ConfusionMatrix(drug['str'].values, pred).plot()
plt.savefig('rfe+en_conf_mat.eps')


# In[ ]:

clf = SGDClassifier(loss='log', penalty='elasticnet', n_jobs=-1, **best_params)
feat_subset = np.arange(rma.shape[1])
for step in np.logspace(4, 0, 4):
    print('Step:', step)
    rfe = RFECV(clf, step=step)
    rfe.fit(rma.iloc[:, feat_subset], drug['str'])
    feat_subset = feat_subset[rfe.support_]
    print('{} features selected'.format(rfe.n_features_))


# In[ ]:

clf = SGDClassifier(loss='log', penalty='elasticnet', n_jobs=-1, **best_params)
feat_subset = np.arange(rma.shape[1])
n_selected = rma.shape[1]
for step in np.logspace(4, 0, 5):
    n_to_select = 
    while n_to_select >= 1:
        rfe = RFE(clf, n_features_to_select=n_to_select, step=step)
        rfe.fit(rma.iloc[:, feat_subset], drug['str'])
        
    
    
    


# In[93]:

'%s'.format('hi')


# In[88]:

print(np.logspace(4, 0, 5))
feat_subset = np.arange(rma.shape[1])
feat_subset


# In[70]:

rma.iloc[:, rfe.support_]


# In[65]:

probes90 = '[1] "1436720_s_at" "1428306_at" "1457425_at" "1437780_at" "1453695_at" [6] "1444249_at" "1439924_x_at" "1437996_s_at" "1441426_at" "1459909_at" [11] "1432573_at" "1417698_at" "1425482_s_at" "1447188_at" "1449386_at" [16] "1429991_at" "1430357_at" "1420239_x_at" "1441500_at" "1425449_at" [21] "1429289_at" "1432995_at" "1439597_at" "1432132_at" "1459719_at" [26] "1442797_x_at" "1415734_at" "1416556_at" "1456834_at" "1425463_at" [31] "1456123_at" "1454193_at" "1457862_at" "1434985_a_at" "1442930_at" [36] "1455302_at" "1431743_a_at" "1454421_at" "1430333_at" "1453336_at" [41] "1445615_at" "1444745_at" "1455909_at" "1438133_a_at" "1425298_a_at" [46] "1449731_s_at" "1432268_at" "1452108_at" "1443977_at" "1458609_at" [51] "1441182_at" "1455660_at" "1440997_at" "1417942_at" "1456501_at" [56] "1450322_s_at" "1431577_at" "1429359_s_at" "1459662_at" "1423009_at" [61] "1439823_at" "1454942_at" "1418027_at" "1431265_at" "1444515_at" [66] "1441778_at" "1457721_at" "1416143_at" "1432621_at" 11435524_at" [71] "1455455_at" "1419831_at" "1424979_at" "1431842_at" "1432495_at" [76] "1437302_at" "1452881_at" "1418201_at" "1458190_at" "1459188_at" [81] "1431905_s_at" "1427194_a_at" "1425172_at" "1457994_at" "1419764_at" [86] "1444816_at" "1440433_at" "1416466_at" "1447348_at" "1455233_at"'.split()
probes90 = [x[1:-1] for x in probes90 if x.endswith('_at"')]
probes90 = set(probes90)


# In[66]:

probes160 = '[1] "1442274 at" "1442797 x at" "1437780 at" "1417942 at" "1437996 s at" [6] "1426234 s at" "1431577 at" "1444249 at" ' '1436720_s_at" "1437626_at" [11] "1453695 at" "1452221 a at" "1459553 at" ' '1445045_at" "1422651_at" [16] "1440445 at" "1459310 at" "1458009 at" ' '1432576 at" "1454942 at" [21] "1439924 x at" "1454320 at" "1427857 x at" "1443268 at" "1439597 at" [26] "1460163 at" "1438133_a_at" "1418361_at" "1455025_at" "1456910 at" [31] "1418555_x_at" "1450322_s_at" "1416039 x at" "1415875 at" "1445682 at" [36] "1459102 at" "1457474_at" "1429236 at" "1456501 at" "1443701 at" [41] "1455302_at" "1429991 at" "1426171 x at" "1444061 at" "1460405_at" [46] "1453540 at" "1438532_at" "1441271 at" "1459909 at" "1449386_at" [51] "1456751_x_at" "1458854_at" "1429398 at" "1445857 at" "1443894 at" [56] "1427738 at" "1419235_s_at" "1446917 at" "1419047 at" "1437922_at" [61] "1419569 a at" "1447237_at" "1424118_a_at" "1437515_at" "1436637 at" [66] "1430357_at" "1450096 at" "1443099 at" "1441500 at" "1443272_at" [71] "1456834 at" "1430090 at" "1418590 at" "1459188 at" "1454303_at" [76] "1459507_at" "1425172 at" "1444453_at" "1431953_at" "1458609_at" [81] "1421921 at" "1426431 at" "1428306_at" "1460273_a_at" "1437572_at" [86] "1444816 at" "1460086 at" "1445735 at" "1432102_at" "1425818 at" [91] "1443124 at" "1442796 at" "1444867 at" "1420729_at" "1432236 a at" [96] "1453743_x_at" "1457946 at" "1416862 at" "1448757 at" "1455233 at" [101] "1438233_at" "1429289 at" "1421362 a at' ' "1446773 at" "1422230 s at" [106] "1446221 at" "1441896 x at" "1419294 at" "1423344 at" "1419166 at" [111] "1442825 at" "1457662 x at" "1440985 at" "1458856_at" "1431952 at" [116] "1437876 at" "1441426 at" "1432867 at" "1440064 at" "1449950 at" [121] "1434892 x at" "1458253 at" "1459643 at" "1446168 at" "1446142 at" [126] "1458698 at" "1438380 at" "1440433 at" "1453859 at" "1432030 at" [131] "1458644 at" "1429492 x at" "1431842 at" "1451613 at" "1431464 a at" [136] "1443987_at" "1449879 at" "1430114 at" "1420368 at" "1451869 at" [141] "1418541_at" "1426952 at" "1446051 at" "1454808 at" "1422262 a at" [146] "1457425_at" "1442177 at" "1416325 at" "1453671 at" "1459128 at" [151] "1449742 at" "1438810 at" "1427499 at" "1454428_at" "1453750 x at" [156] "1451596 a at" "1446370 at" "1437038 x at' ' "1417536 at" "1418953 at"'
probes160 = probes160.replace("' '", '"')
probes160 = probes160.replace(" x at", "_x_at")
probes160 = probes160.replace(" a at", "_a_at")
probes160 = probes160.replace(" s at", "_s_at")
probes160 = probes160.replace(" at", "_at")
probes160 = probes160.split()
probes160 = [x for x in probes160 if not x.startswith('[')]
probes160 = [x.replace('"', "") for x in probes160]
probes160 = set(probes160)


# In[64]:

selected_set = set(rma.columns[support].values)
len(selected_set)


# In[68]:

selected_set & probes160


# In[25]:

foo = np.full((4,), np.nan)
foo[1] = 1
foo[2] = 50
np.nanstd(foo)


# In[ ]:



