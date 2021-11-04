#!/usr/bin/env python3

import argparse, os, glob
from more_itertools import first
import numpy as np
import pandas as pd

from viz import load, analyze_method_dir


def load_record(filename, name=None):
    result = load(filename)
    protocol = first(k.split('_', maxsplit=1)[0] for k in result.keys())
    names = {'R'    : f'{protocol}_recall'            ,
             'ng-R' : f'{protocol}_recall_nogc'       ,
             'zR'   : f'{protocol}_zeroshot_recall'   ,
             'ng-zR': f'{protocol}_ng_zeroshot_recall',
             'mR'   : f'{protocol}_mean_recall'       ,
             'ng-mR': f'{protocol}_ng_mean_recall'    ,}
    return pd.DataFrame({f'{name}@{k}': np.mean(value) * 100
        for name, key in names.items() for k, value in result[key].items()},
            index=None if name is None else [name])


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--precision', type=int, default=1)
    parser.add_argument('-m', '--metrics', nargs='*')
    parser.add_argument('-t', '--threshold')
    parser.add_argument('-T', '--best-threshold')
    parser.add_argument('-n', '--skip-na', action='store_true')
    parser.add_argument('paths', nargs='+')
    args = parser.parse_args()
    if args.best_threshold is not None:
        assert(args.threshold is None)
        args.threshold = args.best_threshold
        args.best_threshold = True
    else:
        args.best_threshold = False

    # Load data.
    dfs = []
    for path in args.paths:
        method_name, _, result_dir = analyze_method_dir(
                path, allow_missing=args.skip_na)
        if result_dir is None:
            continue
        filename = os.path.join(result_dir, 'result_dict.pytorch')
        dfs.append(load_record(filename, name=method_name))
    df = pd.concat(dfs)

    # Format the data.
    if args.threshold is not None:
        assert(args.metrics is None)
        metric = args.threshold
        names, thresholds = zip(
                *[name.rsplit('-', maxsplit=1) for name in df.index])
        assert(all(n == names[0] for n in names[1:]))
        df.index = [t for t in thresholds]
        df = df.sort_values(by=metric, ascending=False)[[metric]]
        if args.best_threshold:
            print(df.index[0])
            return
    elif args.metrics is not None:
        metrics = args.metrics
        if '@' not in metrics:
            metrics = [k for k in df.columns if any(
                k.startswith(m + '@') for m in metrics)]
        df = df[metrics]

    # Print the result.
    pd.set_option('display.max_rows'   , None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', f'{{:.{args.precision}f}}'.format)
    print(df)

if __name__ == '__main__':
    main()
