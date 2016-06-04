# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from hyper.learning import util

import sys
import logging


class NegativeSamplesGenerator(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, Xr, Xe):
        while False:
            yield None

    @abstractproperty
    def nb_sample_sets(self):
        while False:
            yield None


class CorruptedSamplesGenerator(NegativeSamplesGenerator):
    """
    Instances of this class are used for randomly corrupting the subject and objects in
    a set of triples, following the procedure used in [1].

    [1] A Bordes et al. - Translating Embeddings for Modeling Multi-relational Data - NIPS 2013
    """
    def __init__(self, subject_index_generator, subject_candidate_indices,
                 object_index_generator, object_candidate_indices):

        # Generator of random subject indices, and array of candidate indices
        self.subject_index_generator = subject_index_generator
        self.subject_candidate_indices = subject_candidate_indices

        # Generator of random object indices, and array of candidate indices
        self.object_index_generator = object_index_generator
        self.object_candidate_indices = object_candidate_indices

        self._nb_sample_sets = 2

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
        negative_Xr = Xr

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
    def nb_sample_sets(self):
        # TODO: It could be 3 if we also decide to corrupt predicates.
        return self._nb_sample_sets


class LCWANegativeSamplesGenerator(NegativeSamplesGenerator):
    """
    Instances of this class are used for randomly corrupting the objects in a set of triples,
    according to the Local Closed World Assumption [1].

    [1] X L Dong et al. - Knowledge Vault: A Web-Scale Approach to Probabilistic Knowledge Fusion - KDD 2014
    """
    def __init__(self, object_index_generator, object_candidate_indices):

        # Generator of random object indices, and array of candidate indices
        self.object_index_generator = object_index_generator
        self.object_candidate_indices = object_candidate_indices

        self._nb_sample_sets = 1

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
    def nb_sample_sets(self):
        return self._nb_sample_sets


class SchemaAwareNegativeSamplesGenerator(NegativeSamplesGenerator):
    def __init__(self, index_generator, candidate_indices,
                 random_state, predicate2type):
        # Generator of random entity indices, and array of candidate indices
        self.index_generator = index_generator
        self.candidate_indices = candidate_indices

        self._nb_sample_sets = 1

        self.random_state = random_state

        max_index = max(predicate2type.keys())
        self.predicate2type = np.zeros(max_index + 1)
        for predicate_idx, predicate_type in predicate2type.items():
            self.predicate2type[predicate_idx] = predicate_type.value

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
        negative_idxs = self.index_generator(nb_samples, self.candidate_indices)
        negative_Xe = np.copy(Xe)

        for i in range(nb_samples):
            predicate_type = self.predicate2type[Xr[i, 0]]
            if predicate_type == util.PredicateType.one_to_many.value:
                idx_to_corrupt = 0
            elif predicate_type == util.PredicateType.many_to_one.value:
                idx_to_corrupt = 1
            else:
                idx_to_corrupt = self.random_state.randint(0, 2)
            negative_Xe[i, idx_to_corrupt] = negative_idxs[i]

        return [(negative_Xr, negative_Xe)]

    @property
    def nb_sample_sets(self):
        return self._nb_sample_sets


class BernoulliNegativeSamplesGenerator(NegativeSamplesGenerator):
    def __init__(self, index_generator, candidate_indices,
                 random_state, ps_count, po_count):
        # Generator of random entity indices, and array of candidate indices
        self.index_generator = index_generator
        self.candidate_indices = candidate_indices

        self._nb_sample_sets = 1

        self.random_state = random_state

        entities = sorted(set([s for (_, s) in ps_count.keys()] + [o for (_, o) in po_count.keys()]))
        predicates = sorted(set([p for (p, _) in ps_count.keys()] + [p for (p, _) in po_count.keys()]))

        max_entity_index = max(entities)
        max_predicate_index = max(predicates)

        self.objects_per_subject = np.zeros((max_predicate_index + 1, max_entity_index + 1))
        for (predicate_idx, subject_idx), objects_count in ps_count.items():
            self.objects_per_subject[predicate_idx, subject_idx] = objects_count

        self.subjects_per_object = np.zeros((max_predicate_index + 1, max_entity_index + 1))
        for (predicate_idx, object_idx), subjects_count in po_count.items():
            self.objects_per_subject[predicate_idx, object_idx] = subjects_count

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
        negative_idxs = self.index_generator(nb_samples, self.candidate_indices)
        negative_Xe = np.copy(Xe)

        for i in range(nb_samples):
            predicate_idx, subject_idx, object_idx = Xr[i, 0], Xe[i, 0], Xe[i, 1]

            objects_per_subject = self.objects_per_subject[predicate_idx, subject_idx]  # tph
            subjects_per_object = self.subjects_per_object[predicate_idx, object_idx]  # hpt

            # Probability of replacing the subject
            p = objects_per_subject / (objects_per_subject + subjects_per_object)

            idx_to_corrupt = 0 if self.random_state.binomial(1, p) == 1 else 1
            negative_Xe[i, idx_to_corrupt] = negative_idxs[i]

        return [(negative_Xr, negative_Xe)]

    @property
    def nb_sample_sets(self):
        return self._nb_sample_sets


def get_function(function_name):
    this_module = sys.modules[__name__]
    if hasattr(this_module, function_name):
        function = getattr(this_module, function_name)
    else:
        raise ValueError("Unknown negative sampling function: %s" % function_name)
    return function
