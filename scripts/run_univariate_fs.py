import os

QSUB = 'qsub'
CWD = '-cwd'
JOIN = '-j y'
SHELL = '-S /bin/bash'
QUEUE = '-q R4hi.q,R8hi.q,R32hi.q,R128hi.q'
NAME = '-N FS'
OUT = '-o $HOME/bucket/logs/$JOB_ID.txt'

PYTHON = 'python'
SCRIPT = '$HOME/MScThesis/univariate_fs.py'

data_target_tissue = []
epi_ad_tissues = ["cerebellum", '"entorhinal cortex"', '"frontal cortex"',
                  '"superior temporal gyrus"', '"whole blood"']
for tissue in epi_ad_tissues:
    data_target_tissue.append(('epi_ad', 'ad.disease.status', tissue))
    # data_target_tissue.append(('epi_ad', 'braak.stage', tissue))

# for target in ['stress', 'drug']:
#     for data in ['mdd', 'mdd_raw', 'mdd_raw37']:
#         data_target_tissue.append((data, target, None))

for target in ['stress', 'drug']:
    data_target_tissue.append(('mdd_raw37', target, None))


# anova cerebellum

n_folds = '10'
n_iter = 10
test_size = '0.1'
# filters = ['anova', 'infogain_10', 'infogain_exp']
filters = ['chi2']
clfs = ['en', 'svm_linear_kernel', 'svm_linear', 'svm_linear_l1']

jobs = [
    ('mdd_raw37', 'stress', None, 'anova', 'svm_linear_l1'),
    ('mdd_raw37', 'stress', None, 'infogain_10', 'svm_linear_l1'),
    ('mdd_raw37', 'stress', None, 'infogain_exp', 'svm_linear_l1'),
    ('mdd_raw37', 'stress', None, 'infogain_exp', 'svm_linear_kernel'),
    ('mdd_raw_37', 'drug', None, 'anova', 'en'),
    ('mdd_raw_37', 'drug', None, 'anova', 'svm_linear_kernel'),
    ('mdd_raw_37', 'drug', None, 'anova', 'svm_linear_l1'),
    ('mdd_raw_37', 'drug', None, 'infogain_10', 'en'),
    ('mdd_raw_37', 'drug', None, 'infogain_10', 'svm_linear_kernel'),
    ('mdd_raw_37', 'drug', None, 'infogain_10', 'svm_linear_l1'),
    ('mdd_raw_37', 'drug', None, 'infogain_exp', 'en'),
    ('mdd_raw_37', 'drug', None, 'infogain_exp', 'svm_linear_kernel'),
    ('mdd_raw_37', 'drug', None, 'infogain_exp', 'svm_linear_l1')
]

n_jobs = 0

for _ in xrange(n_iter):
    # for data, target, tissue in data_target_tissue:
    for data, target, tissue, filter, clf in jobs:
        # for filter in filters:
        if data == 'epi_ad':
            queue = '-q R32hi.q,R128hi.q,R128.q,R32.q'
        else:
            queue = '-q R4hi.q,R8hi.q,R32hi.q,R128hi.q,R4.q,R8.q,R32.q,R128.q'
        submit_options = [CWD, JOIN, SHELL, queue, NAME, OUT]

        # for clf in clfs:
        command = [
            PYTHON, SCRIPT,
            '--data {0}'.format(data),
            '--target {0}'.format(target),
            '--test-size {0}'.format(test_size),
            '--n-iter {0}'.format('1'),
            '--filter {0}'.format(filter),
            '--clf {0}'.format(clf),
            '-v'
        ]
        if tissue is not None:
            command.append('--tissue {0}'.format(tissue))

        with open('job.sh', 'w') as f:
            f.write('#!/bin/bash\n')
            for option in submit_options:
                f.write('#$ ' + option + '\n')
            f.write('\n' + ' '.join(command) + '\n')

        os.system('qsub job.sh')
        n_jobs += 1

print('# {0} jobs submitted.'.format(n_jobs))
