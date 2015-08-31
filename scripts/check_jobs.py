from __future__ import division, print_function
import json
import os
from os.path import join
from os.path import basename
from subprocess import Popen
from subprocess import PIPE
import re
import shutil

if __name__ == '__main__':

    log_folder = './bucket/logs'
    logs_files = [x for x in os.listdir(log_folder) if re.match('^\d+\.txt', basename(x))]

    qstat = Popen('qstat', stdout=PIPE).communicate()[0]
    running_jobs = [x.split()[0] for x in qstat.split('\n') if 'mouri' in x]

    for log in logs_files:
        job_id = basename(log).rstrip('.txt')
        log = join(log_folder, log)
        print('### {0}'.format(job_id))
        running = False
        if job_id in running_jobs:
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

        if result_file and os.path.isfile(result_file):
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
        else:
            print('No result file found.')

        if not ok and not error and not running and not result_file:
            show = raw_input('Not running and nothing in the log. Show log? y/[n]')
            if show.lower() == 'y':
                with open(log, 'r') as f:
                    for line in f:
                        print(line.rstrip('\n'))
            archive = raw_input('Do you want to archive the log? y/[n]')
            if archive.lower() == 'y':
                shutil.move(log, log.replace(basename(log), 'error_' + basename(log)))
