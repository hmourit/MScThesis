import os

QSUB = 'qsub'
CWD = '-cwd'
JOIN = '-j y'
SHELL = '-S /bin/bash'
QUEUE = '-q R4hi.q,R8hi.q,R32hi.q,R128hi.q'
NAME = '-N RFE'
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
filters = ['anova', 'infogain_10']

for data, target, tissue in data_target_tissue:
    for filter in filters:
        if data == 'epi_ad':
            queue = '-q R32hi.q,R128hi.q'
        else:
            queue = '-q R4hi.q,R8hi.q,R32hi.q,R128hi.q'
        submit_options = [CWD, JOIN, SHELL, queue, NAME, OUT]
        command = [
            PYTHON, SCRIPT,
            '--data {0}'.format(data),
            '--target {0}'.format(target),
            '--test-size {0}'.format(test_size),
            '--n-iter {0}'.format(n_iter),
            '--filter {0}'.format(filter),
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
