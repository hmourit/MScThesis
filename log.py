from __future__ import division, print_function
import json


def save_results(results, folder='../', filename='results.json'):
    with open(folder + filename, 'r') as in_:
        previous_results = json.load(in_)
    with open(folder + filename, 'w') as out:
        results = previous_results.extend(results)
        json.dump(results, out, sort_keys=True, indent=2, separators=(',', ': '))
