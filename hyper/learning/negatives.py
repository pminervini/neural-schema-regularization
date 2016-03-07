# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np


class NegativeSamplesGenerator(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, Xr, Xe):
        while False:
            yield None

    @abstractproperty
    def negatives_per_positive_example(self):
        while False:
            yield None


class CorruptedSamplesGenerator(NegativeSamplesGenerator):
    """
    Instances of this class are used for randomly corrupting the subject and objects in
    a set of triples, following the procedure used in [1].

    [1] A Bordes et al. - Translating Embeddings for Modeling Multi-relational Data - NIPS 2013
    """
    def __init__(self,
                 subject_index_generator, subject_candidate_indices,
                 object_index_generator, object_candidate_indices):

        # Generator of random subject indices, and array of candidate indices
        self.subject_index_generator = subject_index_generator
        self.subject_candidate_indices = subject_candidate_indices

        # Generator of random object indices, and array of candidate indices
        self.object_index_generator = object_index_generator
        self.object_candidate_indices = object_candidate_indices

        self._negatives_per_positive_example = 2

    def __call__(self, Xr, Xe):
        """
        Generates sets of negative examples, by corrupting the facts provided as input.

        :param Xr: [nb_samples, 1] matrix containing the relation indices.
        :param Xe: [nb_samples, 2] matrix containing subject and object indices.
        :return: list of ([nb_samples, 1], [nb_samples, 2]) pairs containing sets of negative examples.
        """
        nb_samples = Xr.shape[0]

        # Relation indices are not changed.
        # TODO: This could be an option anyway - might improve some results.
        negative_Xr = Xr  # np.copy(Xr)

        # Entity (subject and object) indices, on the other hand, are corrupted for generating
        # two new sets of triples.

        # Create a new set of examples by corrupting the subjects
        negative_subject_idxs = self.subject_index_generator(nb_samples, self.subject_candidate_indices)
        negative_Xe_subject = np.copy(Xe)
        negative_Xe_subject[:, 0] = negative_subject_idxs

        # Create a new set of examples by corrupting the objects
        negative_object_idxs = self.object_index_generator(nb_samples, self.object_candidate_indices)
        negative_Xe_object = np.copy(Xe)
        negative_Xe_object[:, 1] = negative_object_idxs

        return [(negative_Xr, negative_Xe_subject), (negative_Xr, negative_Xe_object)]

    @property
    def negatives_per_positive_example(self):
        # TODO: It could be 3 if we also decide to corrupt predicates.
        return self._negatives_per_positive_example


class LCWANegativeSamplesGenerator(NegativeSamplesGenerator):
    """
    Instances of this class are used for randomly corrupting the objects in a set of triples,
    according to the Local Closed World Assumption [1].

    [1] X L Dong et al. - Knowledge Vault: A Web-Scale Approach to Probabilistic Knowledge Fusion - KDD 2014
    """
    def __init__(self,
                 subject_index_generator, subject_candidate_indices,
                 object_index_generator, object_candidate_indices):

        # Generator of random subject indices, and array of candidate indices
        self.subject_index_generator = subject_index_generator
        self.subject_candidate_indices = subject_candidate_indices

        # Generator of random object indices, and array of candidate indices
        self.object_index_generator = object_index_generator
        self.object_candidate_indices = object_candidate_indices

        self._negatives_per_positive_example = 1

    def __call__(self, Xr, Xe):
        """
        Generates sets of negative examples, by corrupting the facts provided as input according to the LCWA.

        :param Xr: [nb_samples, 1] matrix containing the relation indices.
        :param Xe: [nb_samples, 2] matrix containing subject and object indices.
        :return: list of ([nb_samples, 1], [nb_samples, 2]) pairs containing sets of negative examples.
        """
        nb_samples = Xr.shape[0]

        # Relation indices are not changed.
        negative_Xr = np.copy(Xr)

        # Create a new set of examples by corrupting the objects
        negative_object_idxs = self.object_index_generator(nb_samples, self.object_candidate_indices)
        negative_Xe_object = np.copy(Xe)
        negative_Xe_object[:, 1] = negative_object_idxs

        return [(negative_Xr, negative_Xe_object)]

    @property
    def negatives_per_positive_example(self):
        return self._negatives_per_positive_example
