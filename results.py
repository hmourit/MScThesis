from __future__ import division, print_function
import glob
import json
import pandas as pd


def save_results(results, folder='../', filename='results.json'):
    experiment_id = hash(json.dumps(results))
    for result in results:
        result['experiment_id'] = experiment_id
    with open(folder + filename, 'rb') as in_:
        previous_results = json.load(in_)
    with open(folder + filename, 'wb') as out:
        previous_results.extend(results)
        json.dump(previous_results, out, sort_keys=True, indent=2, separators=(',', ': '))


def save_experiment(results, folder='../', filename='result_{}.json'):
    experiment_id = hash(json.dumps(results))
    print('Experiment id:', experiment_id)
    results['experiment_id'] = experiment_id
    with open(folder + filename.format(experiment_id), 'w') as out:
        json.dump(results, out, sort_keys=True, indent=2, separators=(',', ': '))


def flatten_json(data, record_path=None):
    def _pull(data, path, flattened):
        for field in data:
            if isinstance(data[field], dict):
                flattened = _pull(data[field], path + [field], flattened)
            else:
                flattened['.'.join(path + [field])] = data[field]
        return flattened

    if isinstance(data, dict):
        data = [data]

    if record_path is not None:
        if isinstance(record_path, list):
            record_path = '.'.join(record_path)

    new_data = []
    for record in data:
        flattened = _pull(record, [], {})
        if record_path:
            records_value = [x for x in flattened[record_path]]
            del flattened[record_path]
            new_data.extend([dict(flattened, **{record_path: x}) for x in records_value])
        else:
            new_data.append(flattened)
    return new_data


def read_results(files, record_path=None):
    results = [json.load(open(f, 'r')) for f in glob.glob(files)]
    return pd.DataFrame(flatten_json(results, record_path=record_path))
