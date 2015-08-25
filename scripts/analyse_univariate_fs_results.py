from __future__ import division, print_function
import json
import sys
from os.path import join
from Queue import Queue

from glob import glob

methods = ['anova', 'infogain_exp', 'infogain_10', 'rfe']


def extract_json_fields(document):
    q = Queue()
    if isinstance(document, list):
        for d in document:
            q.put(('[', d))
    elif isinstance(document, dict):
        q.put(document)
    else:
        raise ValueError("Document doesn't have a valid format.")

    fields = {}
    while not q.empty():
        prefix, d = q.get()
        if isinstance(d, list):
            for dd in d:
                q.put((prefix + '[', dd))
        elif isinstance(d, dict):
            for key in d:
                q.put((prefix + '.' + key, d[key]))
        else:
            fields.add(prefix)

    return sorted(fields)


def main():
    path = sys.argv[1]
    for method in methods:
        print('### {0}'.format(method))
        files = glob(join(path, '{0}_*.json'.format(method)))
        document = [json.load(f) for f in files]
        for field in extract_json_fields(document):
            print(field)
        print()

if __name__ == '__main__':
    main()
