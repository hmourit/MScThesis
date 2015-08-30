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
    logs = {}
    for log in log_files:
        with open(log, 'r') as f:
            for line in f:
                if line.startswith('Results will be saved to'):
                    result_file = [x for x in line.split() if x.endswith('.json')][0].strip()
                    result_file = os.path.basename(result_file)
                    logs[result_file] = log
                    break

    result_files = sys.argv[1]
    result_files = glob(result_files)

    to_stop = []
    for result in result_files:
        result_basename = os.path.basename(result)
        print('{0}:\t '.format(result_basename), end='', )
        sys.stdout.flush()

        d = json.load(open(result, 'r'))
        last_subset = d['subsets'][-1]['n_features']
        if int(last_subset) >= 1000:
            to_stop.append(logs[result_basename])

        job_id = os.path.basename(logs[result_basename]).rstrip('.txt')

        print('{0}\tJOB_ID: {1}'.format(last_subset, job_id))
        sys.stdout.flush()

    stop = raw_input('Do you want to stop processes after 1000 features? y/[n]')
    if stop.lower() == 'y':
        job_ids = [os.path.basename(logs[x].rstrip('.txt')) for x in to_stop]
        command = ' '.join(['qdel'] + to_stop)
        print(command)
        # for job_id in to_stop:
        #     log_basename = os.path.basename(logs[])