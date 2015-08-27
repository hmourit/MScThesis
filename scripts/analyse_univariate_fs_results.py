from __future__ import division, print_function
import json
import sys
from os.path import join
from os import listdir
import os
from Queue import Queue

from glob import glob

methods = ['anova', 'infogain_exp', 'infogain_10', 'rfe']


def analyse_univariate(document):
    last_it = len(document['experiments'])
    n_subsets = len(document['experiments'[0]])
    last_subset = len(document['experiments'][-1])
    return '{0}:{1}/{2}:{3}'.format(last_it, last_subset, document['n_iter'], n_subsets)


def main():

    f = sys.argv[1]
    f = open(f, 'r')
    for line in f:
        line = line.strip()
        if line.startswith('"iteration'):
            print(line)


# def main():
#     path = sys.argv[1]
#     print(path)
#     for f in listdir(path):
#         basename = os.path.basename(f)
#         if basename.startswith(('anova', 'infogain')):
#             try:
#                 status = analyse_univariate(json.load(open(join(path, f), 'r')))
#             except:
#                 status = 'ERROR'
#             print('{0}\t{1}'.format(basename, status))


    # for method in methods:
    #     print('### {0}'.format(method))
    #     files = glob(join(path, '{0}_*.json'.format(method)))
    #     document = [json.load(f) for f in files]
    #     for field in extract_json_fields(document):
    #         print(field)
    #     print()

if __name__ == '__main__':
    main()
