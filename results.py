from __future__ import division, print_function
import json


def save_results(results, folder='../', filename='results.json'):
    experiment_id = hash(json.dumps(results))
    for result in results:
        result['experiment_id'] = experiment_id
    with open(folder + filename, 'rb') as in_:
        previous_results = json.load(in_)
    with open(folder + filename, 'wb') as out:
        previous_results.extend(results)
        json.dump(previous_results, out, sort_keys=True, indent=2, separators=(',', ': '))
