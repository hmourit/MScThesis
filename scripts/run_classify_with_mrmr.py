from __future__ import division, print_function
import os
import shutil
from os.path import join, basename

if __name__ == '__main__':
    QSUB = 'qsub'
    CWD = '-cwd'
    JOIN = '-j y'
    SHELL = '-S /bin/bash'
    QUEUE = '-q R32hi.q,R128hi.q,R128.q,R32.q'
    NAME = '-N FS'
    OUT = '-o $HOME/bucket/logs/$JOB_ID.txt'

    PYTHON = 'python'
    SCRIPT = '$HOME/MScThesis/classify_with_mrmr.py'

    submit_options = [CWD, JOIN, SHELL, QUEUE, NAME, OUT]

    clfs = ['en', 'svm_linear_kernel', 'svm_linear_l1']

    n_jobs = 0

    mrmr_result = None

    mrmr_results_path = './bucket/results/finished_mrmr'
    results = os.listdir(mrmr_results_path)
    results = [join(mrmr_results_path, x) for x in results if x.startswith('subsets_mrmr')]

    used = []
    with open(join(mrmr_results_path, 'used.txt'), 'r') as f:
        for line in f:
            used.append(line.rstrip('\n'))

    for mrmr_result in results:
        if mrmr_result in used:
            print('{0} already used'.format(basename(mrmr_result)))
            continue
        for clf in clfs:
            command = [
                PYTHON, SCRIPT,
                '--test-size 0.1',
                '--mrmr-result {0}'.format(mrmr_result),
                '--clf {0}'.format(clf),
                '-v'
            ]

            print(command)

            # with open('job.sh', 'w') as f:
            #     f.write('#!/bin/bash\n')
            #     for option in submit_options:
            #         f.write('#$ ' + option + '\n')
            #     f.write('\n' + ' '.join(command) + '\n')
            #
            # os.system('qsub job.sh')
            n_jobs += 1

        # with open(join(mrmr_results_path, 'used.txt'), 'a') as f:
        #     f.write(mrmr_result + '\n')

    print('# {0} jobs submitted.'.format(n_jobs))