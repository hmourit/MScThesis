from __future__ import division, print_function
from Queue import Queue

from datetime import datetime
import json
import cPickle as pickle

RESULTS_FILE = '../results'


def log_results(log):
    with open(RESULTS_FILE, 'r') as in_:
        logs = json.load(in_)
    with open(RESULTS_FILE, 'w') as out:
        logs.append(log)
        json.dump(logs, out, sort_keys=True, indent=2, separators=(',', ': '))


class Timer():
    def __init__(self):
        self.start = datetime.now()

    def elapsed(self):
        return str(datetime.now() - self.start)[:-7]


def load_data():
    with open('rma.pickle', 'rb') as in_:
        rma = pickle.load(in_)
    with open('drug.pickle', 'rb') as in_:
        drug = pickle.load(in_)
    with open('stress.pickle', 'rb') as in_:
        stress = pickle.load(in_)

    return rma, drug, stress


# def extract_json_fields():
#     def _extract_fields(d, path):
#         fields = set()
#         for k in d:
#             if isinstance(d[k], dict):
#                 fields |= _extract_fields(d[k], path + k + '.')
#             else:
#                 fields.add(path + k)
#         return fields
#
#
#     fields = set()
#     for f in files:
#         results = json.load(open(f, 'rb'))
#         if isinstance(results, dict):
#             results = [results]
#         for experiment in results:
#             fields |= _extract_fields(experiment, '')
#
#     return sorted(fields)

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
