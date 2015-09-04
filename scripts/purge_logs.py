#!/bin/python

import sys
import shutil
import os
import re

TRACEBACK = "Traceback (most recent call last):"
RESULTS_WRITTEN = "# Results written to "
OK = '# OK'


def main():
    logs_path = sys.argv[1]
    logs = os.listdir(logs_path)
    for log_file in logs:
        if not re.match("\d+\.txt$", os.path.basename(log_file)):
            continue
        lines = None
        with open(log_file, 'r') as f:
            lines = f.readlines()

        state = None
        if lines[-1].startswith(OK):
            state = 'finished_OK'
        elif lines[-1].startswith(RESULTS_WRITTEN):
            state = 'finished_add_OK'
        elif any(x.startswith(TRACEBACK) for x in lines):
            state = 'error'
        else:
            print "I don't know what to do with %s" % os.path.basename(log_file)
            continue

        basename = os.path.basename(log_file)
        new_basename = None
        if state == 'finished_OK':
            new_basename = 'finished_' + basename
        elif state == 'finished_add_OK':
            with open(log_file, 'a') as f:
                f.write(OK + '\n')
            new_basename = 'finished_' + basename
        elif state == 'error':
            new_basename = 'error_' + basename
        else:
            print "Unknown state"
            continue
        shutil.move(log_file, log_file.replace(basename, new_basename))
        print '%s moved to %s' % (basename, new_basename)


if __name__ == '__main__':
    main()
