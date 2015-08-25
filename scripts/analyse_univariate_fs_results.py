from __future__ import division, print_function
import json

from utils import extract_json_fields
from glob import glob

methods = {'anova', 'infogain_exp', 'infogain_10', 'rfe'}

for method in methods:
    print('### {}'.format(method))
    files = glob('{}_*.json'.format(method))
    document = [json.load(f) for f in files]
    for field in extract_json_fields(document):
        print(field)
    print()
