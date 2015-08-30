from __future__ import division, print_function
from glob import glob
import json
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: COMMAND results_pattern logs_pattern')

    log_files = sys.argv[2]
    log_files = glob(log_files)
    print('Retrieving JOB_IDs...')
    ids = {}
    for log in log_files:
        job_id = os.path.basename(log).rstrip('.txt')
        with open(log, 'r') as f:
            for line in f:
                if line.startswith('Results will be saved to'):
                    result_file = [x for x in line.split() if x.endswith('.json')][0].strip()
                    result_file = os.path.basename(result_file)
                    ids[result_file] = job_id
                    break

    result_files = sys.argv[1]
    result_files = glob(result_files)

    for result in result_files:
        result_basename = os.path.basename(result)
        print('{0}:\t '.format(result_basename), end='', )
        sys.stdout.flush()

        d = json.load(open(result, 'r'))
        last_subset = d['subsets'][-1]['n_features']

        print('{0}\tJOB_ID: {1}'.format(last_subset, ids[result_basename]))
        sys.stdout.flush()
