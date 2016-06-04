#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io as sio
from sklearn.manifold import MDS, TSNE

import hyper

import sys
import logging

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2016'


def main(argv):
    path = '/home/pasquale/insight/workspace_ibm/data/kinases/mtx/' \
           'reproduction_pre-2003_top-1600_trimmed/entity_centroids.mat'
    entity_centroids_dict = sio.loadmat(path)

    sparse_matrix = entity_centroids_dict['entity vector space']

    X = sparse_matrix.todense()

    #mds = MDS(n_components=2)
    #Y = mds.fit_transform(X)

    tsne = TSNE(n_components=2)
    Y = tsne.fit_transform(X)

    import matplotlib.pyplot as plt

    plt.plot(Y[:, 0], Y[:, 1], 'ro')
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
