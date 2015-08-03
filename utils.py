from __future__ import division, print_function

from datetime import datetime
import json
import cPickle as pickle
# import matplotlib.pyplot as plt

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

# def plot_with_err(x, data, **kwargs):
#     mu, std = data.mean(1), data.std(1)
#     lines = plt.plot(x, mu, '-', **kwargs)
#     plt.fill_between(x, mu - std, mu + std, edgecolor='none',
#                      facecolor=lines[0].get_color(), alpha=0.2)
