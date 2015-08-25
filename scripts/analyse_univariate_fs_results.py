from __future__ import division, print_function
import json
import sys
from os.path import join
from Queue import Queue

from glob import glob

methods = ['anova', 'infogain_exp', 'infogain_10', 'rfe']

def main():
    path = sys.argv
    print(path)
    # for method in methods:
    #     print('### {0}'.format(method))
    #     files = glob(join(path, '{0}_*.json'.format(method)))
    #     document = [json.load(f) for f in files]
    #     for field in extract_json_fields(document):
    #         print(field)
    #     print()

if __name__ == '__main__':
    main()
