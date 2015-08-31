from __future__ import division, print_function
import json
import os
from os.path import basename
import subprocess
import re
import shutil

if __name__ == '__main__':

    logs_files = [x for x in os.listdir('./bucket/logs') if re.match('^\d+\.txt', basename(x))]

    running = [x.split()[0] for x in subprocess.check_output('qstat').split('\n') if 'mouri' in x]

    for log in logs_files:
        job_id = basename(log).rstrip('.txt')
        print('### {0}'.format(job_id))
        running = False
        if job_id in running:
            running = True

        result_file = None
        ok = False
        error = False
        with open(log, 'r') as f:
            for line in f:
                if line.startswith('Results will be saved to'):
                    result_file = [x for x in line.split() if x.endswith('.json')][0].strip()
                elif line.startswith('# OK'):
                    ok = True
            if 'Err' in line.lower():
                error = True
                print(line)

        try:
            _ = json.load(open(result_file, 'r'))
        except ValueError as e:
            print(e)
            remove = raw_input('Do you want to archive log, remove result and stop job? y/[n]')
            if remove.lower() == 'y':
                if running:
                    os.system('qdel {0}'.format(job_id))
                shutil.move(log, log.replace(basename(log), 'error_' + basename(log)))
                os.remove(result_file)

        if not ok and not error and not running and not result_file:
            show = raw_input('Not running and nothing in the log. Show log? y/[n]')
            if show.lower() == 'y':
                with open(log, 'r') as f:
                    for line in f:
                        print(line)
            archive = raw_input('Do you want to archive the log? y/[n]')
            if archive.lower() == 'y':
                shutil.move(log, log.replace(basename(log), 'error_' + basename(log)))
