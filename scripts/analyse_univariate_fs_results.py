from __future__ import division, print_function
import json
import sys
from os.path import join

from utils import extract_json_fields
from glob import glob

methods = {'anova', 'infogain_exp', 'infogain_10', 'rfe'}


def main():
    path = sys.argv[0]
    for method in methods:
        print('### {}'.format(method))
        files = glob(join(path, '{}_*.json'.format(method)))
        document = [json.load(f) for f in files]
        for field in extract_json_fields(document):
            print(field)
        print()

if __name__ == '__main__':
    main()
