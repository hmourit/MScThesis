from __future__ import division, print_function
import argparse
import os
from results import compact_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Task to be done.')
    parser.add_argument('--wd', help='Specify working directory')
    parser.add_argument('-r', '--results',
                        help='Results directory.')
    args = parser.parse_args()

    if args.wd:
        os.chdir(args.wd)

    if args.task == 'compact-results':
        compact_results(results_path=args.results, verbose=True)


if __name__ == '__main__':
    main()
