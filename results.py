from __future__ import division, print_function
from glob import glob
import json
import os
import zipfile
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


def save_experiment(results, folder='../', filename='result_{}.json', verbose=False, error=False):
    experiment_id = hash(json.dumps(results))
    if verbose:
        print('Experiment id:', experiment_id)
    results['experiment_id'] = experiment_id
    if filename is not None:
        if error:
            filename = 'error_' + filename
        with open(os.path.join(folder, filename.format(experiment_id)), 'w') as out:
            json.dump(results, out, sort_keys=True, indent=2, separators=(',', ': '))
    else:
        print(json.dumps(results, sort_keys=True, indent=2, separators=(',', ': ')))


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
    results = [json.load(open(f, 'r')) for f in glob(files)]
    return pd.DataFrame(flatten_json(results, record_path=record_path))


def compact_results(
        results_path='./results',
        result_file='result_*.json',
        block_file='results_block_*.json',
        zipped_file='zipped_results.zip',
        max_block_size=1,
        verbose=False):

    def _zip_and_remove(to_remove, results_path, zipped_file, verbose):
        for f in to_remove:
            if verbose:
                print('Adding {} to {} and removing.'
                      .format(os.path.basename(f), zipped_file))
            with zipfile.ZipFile(os.path.join(results_path, zipped_file), 'a') as z:
                z.write(f)
            os.remove(f)

    def _select_block(results_path, block_file, max_block_size):
        blocks = sorted(glob(os.path.join(results_path, block_file)))

        if len(blocks) > 0:
            current_block = blocks[-1]
            block_size = os.path.getsize(current_block)
        else:
            current_block = ''
            block_size = max_block_size * 1024 * 1024 + 1

        if block_size / 1024 / 1024 > max_block_size:
            new_block_name = block_file.replace('*', '{:04d}'.format(len(blocks)))
            current_block = os.path.join(results_path, new_block_name)
            block_size = 0
            results = []
            if verbose:
                msg = 'Starting new block: {}'.format(os.path.basename(current_block))
                print('{fill:{fill}^{len}}\n{0}\n{fill:{fill}^{len}}'
                      .format(msg, fill='-', len=len(msg)))
        else:
            with open(current_block, 'r') as in_:
                results = json.load(in_)
            if verbose:
                msg = 'Continuing block: {}'.format(os.path.basename(current_block))
                print('{fill:{fill}^{len}}\n{0}\n{fill:{fill}^{len}}'
                      .format(msg, fill='-', len=len(msg)))

        return current_block, results, block_size

    def _write_block(block, results, verbose):
        with open(block, 'w') as out:
            json.dump(results, out, sort_keys=True, indent=2, separators=(',', ': '))
        if verbose:
            msg = 'Writing block: {}'.format(os.path.basename(block))
            print('{fill:{fill}^{len}}\n{0}\n{fill:{fill}^{len}}'
                  .format(msg, fill='-', len=len(msg)))

    current_block, results, block_size = _select_block(results_path, block_file, max_block_size)
    to_remove = []
    for f in glob(os.path.join(results_path, result_file)):
        if verbose:
            print('Adding file: {}'.format(os.path.basename(f)))
        block_size += os.path.getsize(f)
        with open(f, 'r') as in_:
            experiment = json.load(in_)
        if isinstance(experiment, dict):
            experiment = [experiment]
        results.extend(experiment)
        to_remove.append(f)

        if block_size / 1024 / 1024 > max_block_size:
            _write_block(current_block, results, verbose)
            print(zipped_file)
            _zip_and_remove(to_remove, results_path, zipped_file, verbose)
            to_remove = []
            current_block, results, block_size = _select_block(results_path, block_file,
                                                               max_block_size)

    _write_block(current_block, results, verbose)
    _zip_and_remove(to_remove, results_path, zipped_file, verbose)


def apply_transformation(result_files, transform, verbose=False):

    if isinstance(result_files, str):
        if '*' in result_files or '?' in result_files:
            result_files = glob(result_files)
        else:
            result_files = [result_files]

    for file_ in result_files:
        if verbose:
            print('*** {}'.format(os.path.basename(file_)))
        results = json.load(open(file_, 'r'))
        if isinstance(results, dict):
            results = [results]

        modified = 0
        total = 0
        for experiment in results:
            total += 1
            if transform(experiment):
                modified += 1

        if verbose:
            print('{}/{} experiments modified.'.format(modified, total))

        if modified > 0:
            if len(results) == 1:
                results = results[0]

            with open(file_, 'w') as out:
                json.dump(results, out)
