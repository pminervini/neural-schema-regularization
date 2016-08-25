#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import seaborn as sns

import h5py
import pickle

import sys
import logging
import argparse

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2016'


def save_heatmap(model, M, column_names, path, vmin=None, vmax=None, annot_size=6, kb_name=None):
    #sns.set(font='monospace')

    rc = {}
    if kb_name in ['M', 'Y']:
        rc = {'axes.labelsize': 16, 'axes.titlesize': 16,
              'xtick.labelsize': 16, 'ytick.labelsize': 16}

    #sns.plotting_context(context='paper')

    sns.set(rc=rc)
    sns.set_context(rc=rc)
    #plt.rc('savefig', dpi=100)

    name_to_abbr = {
        '<http://dbpedia.org/ontology/musicalArtist>': 'musical arist',
        '<http://dbpedia.org/ontology/musicalBand>': 'musical band',
        '<http://dbpedia.org/ontology/associatedMusicalArtist>': 'associated musical arist',
        '<http://dbpedia.org/ontology/associatedBand>': 'associated band'
    }

    def name_transformer(name):
        res = name_to_abbr[name] if name in name_to_abbr else name
        if res.startswith('_'):
            res = res[1:].replace('_', ' ')
        return res

    df = pd.DataFrame(M)
    df.index = [name_transformer(e) for e in column_names]

    f, ax = plt.subplots()
    #f.set_tight_layout(True)

    sns.heatmap(df, linewidths=.0, annot=True, fmt='.1f', annot_kws={'size': annot_size},
                xticklabels=['' for _ in range(M.shape[0])], square=True, robust=True,
                cbar=False, vmin=vmin, vmax=vmax, ax=ax)

    xlabel = ''

    if model in ['ComplEx']:
        idx = int(M.shape[1] / 2)
        ax.axvline(idx, c='w')
        xlabel += '       Real Part                 Imaginary Part  '
    ax.set(xlabel=xlabel, ylabel='Predicates')

    for i in range(M.shape[0]):
        if (i > 0 and i % 2 == 0 and kb_name in ['WN', 'M']) or (i in [1] and kb_name == 'WN') or (i in [1, 2, 3] and kb_name == 'Y'):
            ax.axhline(i, c='w')

    ax.xaxis.set_label_position('top')

    f.tight_layout()
    plt.savefig(path, bbox_inches='tight')


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Model Weights Explorer', formatter_class=formatter)

    argparser.add_argument('weights', action='store', type=str, default=None)
    argparser.add_argument('parser', action='store', type=str, default=None)

    argparser.add_argument('--entities', '-e', action='store', nargs='+', default=[])
    argparser.add_argument('--predicates', '-p', action='store', nargs='+', default=[])

    argparser.add_argument('--model', action='store', type=str, default=None)
    argparser.add_argument('--heatmap', action='store', type=str, default=None)

    argparser.add_argument('--vmin', action='store', type=float, default=None)
    argparser.add_argument('--vmax', action='store', type=float, default=None)

    argparser.add_argument('--annot-size', action='store', type=int, default=None)
    argparser.add_argument('--kb-name', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    weights_path = args.weights
    parser_path = args.parser

    entities = args.entities
    predicates = args.predicates

    model = args.model
    heatmap_path = args.heatmap

    vmin = args.vmin
    vmax = args.vmax
    annot_size = args.annot_size
    kb_name = args.kb_name

    with h5py.File(weights_path) as f:
        E = f['/embedding_2/embedding_2_W'][()]
        W = f['/embedding_1/embedding_1_W'][()]

    with open(parser_path, 'rb') as f:
        parser = pickle.load(f)

    entity_index = parser.entity_index
    predicate_index = parser.predicate_index

    entity_idx_lst = [entity_index[e] for e in entities]
    predicate_idx_lst = [predicate_index[p] for p in predicates]

    if len(entity_idx_lst) > 0:
        print('# Entities')

        for i in range(E.shape[1]):
            print('[%3i]\t%s' % (i, '\t'.join(["%.3f" % E[e_idx][i] for e_idx in entity_idx_lst])))

        if heatmap_path is not None:
            M = E[np.array(entity_idx_lst), :]
            save_heatmap(model=model, M=M, column_names=entities, path=heatmap_path,
                         vmin=vmin, vmax=vmax, annot_size=annot_size, kb_name=kb_name)

    if len(predicate_idx_lst) > 0:
        print('# Predicates')

        for i in range(W.shape[1]):
            print('[%3i]\t%s' % (i, '\t'.join(["%.3f" % W[p_idx][i] for p_idx in predicate_idx_lst])))

        if heatmap_path is not None:
            M = W[np.array(predicate_idx_lst), :]
            save_heatmap(model=model, M=M, column_names=predicates, path=heatmap_path,
                         vmin=vmin, vmax=vmax, annot_size=annot_size, kb_name=kb_name)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
