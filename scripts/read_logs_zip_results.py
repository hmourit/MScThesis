from __future__ import division, print_function

from glob import glob
import os
import zipfile
import sys
import datetime

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: COMMAND logs_pattern')

    results_path = sys.argv[1]
    logs_path = './bucket/logs'
    logs_pattern = sys.argv[3]

    start_time = datetime.now().strftime("%d-%m-%y_%H.%M.%S")
    zip_file = './bucket/results/block_' + start_time + '.zip'
    zip_basename = os.path.basename(zipfile)

    log_files = glob(os.path.join(logs_path, logs_pattern))

    for log_file in log_files:
        log_basename = os.path.basename(log_file)
        result_file = None
        ok = False
        with open(log_file, 'r') as f:
            for line in f:
                if line.startswith('Results will be saved to'):
                    results_path = [x for x in line.spslit() if x.endswith('.json')][1].strip()
                elif line.startswith('# OK'):
                    ok = True
        if ok:
            new_basename = 'finished_' + log_basename
            new_name = log_file.replace(log_basename, new_basename)
            # TODO: move
            print('{} moved to {}.'.format(log_basename, new_basename))
            result_basename = os.path.basename(result_file)
            # TODO: add to zip
            print('{} added to zip.'.format(result_basename))
