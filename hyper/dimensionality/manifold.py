# -*- coding: utf-8 -*-

import  abc

import sklearn
import sklearn.manifold
import sklearn.datasets


class DimensionalityReductionMethod(abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, X): pass


class TSNE(DimensionalityReductionMethod):
    def __init__(self, n_components=2):
        self.tsne = sklearn.manifold.TSNE(n_components=n_components)

    def __call__(self, X):
        return self.tsne.fit_transform(X)


class MDS(DimensionalityReductionMethod):
    def __init__(self, n_components=2):
        self.mds = sklearn.manifold.MDS(n_components=n_components)

    def __call__(self, X):
        return self.mds.fit_transform(X)


class ISOMAP(DimensionalityReductionMethod):
    def __init__(self, n_neighbors=5, n_components=2):
        self.isomap = sklearn.manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)

    def __call__(self, X):
        return self.isomap.fit_transform(X)


class LLE(DimensionalityReductionMethod):
    def __init__(self, n_neighbors=5, n_components=2):
        self.lle = sklearn.manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)

    def __call__(self, X):
        return self.lle.fit_transform(X)
