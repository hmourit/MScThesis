import os

QSUB = 'qsub'
CWD = '-cwd'
JOIN = '-j y'
SHELL = '-S /bin/bash'
QUEUE = '-q R4hi.q,R8hi.q,R32hi.q,R128hi.q'
NAME = 'RFE'
OUT = '-o $HOME/bucket/logs/$JOB_ID.txt'

PYTHON = 'python'
SCRIPT = '$HOME/MScThesis/rfe2.py'

data_target_tissue = []
epi_ad_tissues = ["cerebellum", "entorhinal cortex", "frontal cortex", "superior temporal gyrus",
                  "whole blood"]
for tissue in epi_ad_tissues:
    data_target_tissue.append(('epi_ad', 'ad.disease.status', tissue))

for target in ['stress', 'drug']:
    for data in ['mdd', 'mdd_raw', 'mdd_raw37']:
        data_target_tissue.append((data, target, None))


n_folds = '10'
n_iter = '10'
test_size = '0.1'
clfs = ['en', 'svm_linear_kernel', 'svm_linear', 'svm_linear_l1']

for data, target, tissue in data_target_tissue:
    for clf in clfs:
        if data == 'epi_ad':
            queue = '-q R32hi.q,R128hi.q'
        else:
            queue = '-q R4hi.q,R8hi.q,R32hi.q,R128hi.q'
        command = [
            QSUB, CWD, JOIN, SHELL, QUEUE, NAME, OUT, PYTHON, PYTHON, SCRIPT,
            '--data {}'.format(data),
            '--target {}'.format(target),
            '--clf {}'.format(clf),
            '--test-size {}'.format(test_size),
            '--n-iter {}'.format(n_iter),
            '--n-folds {}'.format(n_folds),
            '-v'
        ]
        if tissue is not None:
            command.append('--tissue {}'.format(tissue))

        os.system(' '.join(command))
