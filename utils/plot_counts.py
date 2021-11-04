#!/usr/bin/env python3

import argparse, os, ast
import pandas as pd
from matplotlib import pyplot as plt


translation = {'obj': 'object', 'rel': 'relationship'}

def split_multiindex_df(df, level, axis):
    return [(k, d.xs(k, level=level, axis=axis))
            for k, d in df.groupby(level=level, axis=axis)]

def get_lowest_uncommon_component(paths, omit_ext=False):
    ps = paths
    if omit_ext:
        ps = [os.path.splitext(p)[0] if not os.path.isdir(p) else p for p in ps]
            # remove extension only from the lowest level
    while not any(p == '' for p in ps):
        bs = [os.path.basename(p) for p in ps]
        if not all(b == bs[0] for b in bs):
            return bs
        ps = [os.path.dirname(p) for p in ps]
    return ps


def int_tuple(s):
    return tuple(int(x) for x in ast.literal_eval(s))

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--figsize', type=int_tuple)
    parser.add_argument('--dpi', type=int)
    parser.add_argument('-K', '--top-k', default=100)
    parser.add_argument('-O', '--output')
    parser.add_argument('-n', '--names', nargs='+')
    parser.add_argument('paths', nargs='+')
    args = parser.parse_args()
    names = get_lowest_uncommon_component(args.paths, omit_ext=True) \
        if args.names is None else args.names
    paths = args.paths
    if os.path.isdir(os.path.join(args.paths[0])):
        paths = [os.path.join(p, 'eval', f'{args.top_k}.pkl') for p in paths]
    assert(len(names) == len(paths))

    # Load and format data.
    data = {}
    for path in paths:
        for k, df in split_multiindex_df(pd.read_pickle(path), 0, 1):
            data.setdefault(k, []).append(df)
    for k, dfs in data.items():
        df = pd.concat([df.unstack() for df in dfs], axis=1).T
        df.index = names
        data[k] = df

    # Plot the data.
    for k, df in data.items(): # for objs and rels
        plt.figure(figsize=args.figsize, dpi=args.dpi)
        bottom = None
        for l, vs in df.iteritems():
            plt.bar(vs.index, vs.values, label=f'{l[0]}-{l[1]}', bottom=bottom)
            bottom = vs.to_numpy() if bottom is None else bottom + vs.to_numpy()
        plt.ylabel(f'The number of {translation[k]}s per image')
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

        # Show or save the plot.
        plt.tight_layout()
        if args.output is None:
            plt.show()
        else:
            if args.output.format('') == args.output:
                stem, ext = os.path.splitext(args.output)
                args.output = f'{stem}-{{}}{ext}'
            plt.savefig(args.output.format(k))

if __name__ == '__main__':
    main()
