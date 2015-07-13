from __future__ import division, print_function

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