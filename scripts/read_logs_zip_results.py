from __future__ import division, print_function

from glob import glob
import os
import zipfile
import sys
from datetime import datetime
import shutil

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: COMMAND logs_pattern')

    logs_path = './bucket/logs'
    logs_pattern = sys.argv[1]

    start_time = datetime.now().strftime("%y%m%d_%H%M%S")
    zip_file = './bucket/results/block_' + start_time + '.zip'
    zip_basename = os.path.basename(zip_file)

    logs_pattern = os.path.join(logs_path, logs_pattern)
    print('Reading {0}'.format(logs_pattern))
    log_files = glob(logs_pattern)

    any_finished = False

    for log_file in log_files:
        log_basename = os.path.basename(log_file)
        print('{0}: '.format(log_basename), end='', )
        result_file = None
        ok = False
        error = False
        with open(log_file, 'r') as f:
            for line in f:
                if line.startswith('Results will be saved to'):
                    result_file = [x for x in line.split() if x.endswith('.json')][0].strip()
                elif line.startswith('# OK'):
                    ok = True
            if 'Error' in line:
                print(line)
                error = True
        if ok:
            new_basename = 'finished_' + log_basename
            new_name = log_file.replace(log_basename, new_basename)
            print('Finished')
            result_basename = os.path.basename(result_file)
            # z = zipfile.ZipFile(zip_file, 'a', compression=zipfile.ZIP_DEFLATED)
            # try:
            #     z.write(result_file)
            #     shutil.move(log_file, new_name)
            #     os.remove(result_file)
            #     any_finished = True
            #     print('-: {0} added to zip and removed.'.format(result_basename))
            # except OSError as e:
            #     print(e)
            # finally:
            #     z.close()
        elif error:
            new_basename = 'finished_' + log_basename
            new_name = log_file.replace(log_basename, new_basename)
            result_basename = os.path.basename(result_file)
            shutil.move(log_file, new_name)
            remove = raw_input('Remove results? y/[n]')
            if remove:
                if os.path.isfile(result_file):
                    os.remove(result_file)
                else:
                    print("Couldn't remove " + result_file)
        else:
            print('Not finished yet')

    if any_finished:
        print('Results zipped in {0}'.format(zip_basename))
