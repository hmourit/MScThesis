from __future__ import division, print_function
import os

if __name__ == '__main__':
    QSUB = 'qsub'
    CWD = '-cwd'
    JOIN = '-j y'
    SHELL = '-S /bin/bash'
    QUEUE = '-q R4hi.q,R8hi.q,R32hi.q,R128hi.q'
    NAME = '-N FS'
    OUT = '-o $HOME/bucket/logs/$JOB_ID.txt'

    PYTHON = 'python'
    SCRIPT = '$HOME/MScThesis/mrmr.py'

    epi_ad_tissues = ["cerebellum", '"entorhinal cortex"', '"frontal cortex"',
                      '"superior temporal gyrus"', '"whole blood"']

    data_target_tissue = []
    for tissue in epi_ad_tissues:
        data_target_tissue.append(('epi_ad', 'ad.disease.status', tissue))

    for target in ['stress', 'drug']:
        data_target_tissue.append(('mdd_raw37', target, None))

    n_iter = 10
    test_size = '0.1'

    n_jobs = 0
    for _ in xrange(n_iter):
        for data, target, tissue in data_target_tissue:
            if data == 'epi_ad':
                queue = '-q R32hi.q,R128hi.q,R128.q,R32.q'
            else:
                queue = '-q R4hi.q,R8hi.q,R32hi.q,R128hi.q,R4.q,R8.q,R32.q,R128.q'
            submit_options = [CWD, JOIN, SHELL, queue, NAME, OUT]

            command = [
                PYTHON, SCRIPT,
                '--data {0}'.format(data),
                '--target {0}'.format(target),
                '--test-size {0}'.format(test_size),
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