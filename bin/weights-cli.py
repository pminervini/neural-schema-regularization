#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns

import h5py
import pickle

import sys
import logging
import argparse

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2016'


def save_heatmap(model, M, column_names, path):
    sns.set(font="monospace")

    df = pd.DataFrame(M)
    df.index = column_names

    f, ax = plt.subplots()

    # Create a custom colormap for the heatmap values
    cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)

    ax = sns.heatmap(df, linewidths=0.0, cmap=cmap, xticklabels=False, square=True)

    xlabel = 'Embeddings'

    if model in ['ComplEx']:
        idx = int(M.shape[1] / 2)
        ax.axvline(idx, c='w')

        xlabel += ' (Real and Imaginary Part)'
    ax.set(xlabel=xlabel, ylabel='')

    for i in range(M.shape[0]):
        if i > 0 and i % 2 == 0:
            ax.axhline(i, c='w')

    #ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(base=idx))

    plt.yticks(rotation=0)

    plt.figure(figsize=(45, 10))
    f.savefig(path)


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

    args = argparser.parse_args(argv)

    weights_path = args.weights
    parser_path = args.parser

    entities = args.entities
    predicates = args.predicates

    model = args.model
    heatmap_path = args.heatmap

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
            save_heatmap(model=model, M=M, column_names=entities, path=heatmap_path)

    if len(predicate_idx_lst) > 0:
        print('# Predicates')

        for i in range(W.shape[1]):
            print('[%3i]\t%s' % (i, '\t'.join(["%.3f" % W[p_idx][i] for p_idx in predicate_idx_lst])))

        if heatmap_path is not None:
            M = W[np.array(predicate_idx_lst), :]
            save_heatmap(model=model, M=M, column_names=predicates, path=heatmap_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
