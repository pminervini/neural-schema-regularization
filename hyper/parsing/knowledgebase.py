# -*- coding: utf-8 -*-

import logging


class Fact(object):
    def __init__(self, predicate_name, argument_names):
        self.predicate_name = predicate_name
        self.argument_names = argument_names


class KnowledgeBaseParser(object):
    def __init__(self, facts, entity_partial_ordering=None, predicate_partial_ordering=None):
        self.entity_vocabulary, self.predicate_vocabulary = set(), set()
        self.entity_index, self.predicate_index = dict(), dict()

        for fact in facts:
            self.predicate_vocabulary.add(fact.predicate_name)
            for arg in fact.argument_names:
                self.entity_vocabulary.add(arg)

        self._fit(entity_partial_ordering=entity_partial_ordering,
                  predicate_partial_ordering=predicate_partial_ordering)

    def _fit(self, entity_partial_ordering=None, predicate_partial_ordering=None):
        """
        Required before using facts_to_sequences
        :param facts: List or generator of facts.
        :return:
        """
        if entity_partial_ordering is not None:
            sorted_entity_lst = sorted(entity_partial_ordering, key=entity_partial_ordering.get, reverse=True)
        else:
            sorted_entity_lst = sorted(self.entity_vocabulary)

        if predicate_partial_ordering is not None:
            sorted_predicate_lst = sorted(predicate_partial_ordering, key=predicate_partial_ordering.get, reverse=True)
        else:
            sorted_predicate_lst = sorted(self.predicate_vocabulary)

        self.entity_index = {entity: idx for idx, entity in enumerate(sorted_entity_lst, start=1)}
        self.predicate_index = {predicate: idx for idx, predicate in enumerate(sorted_predicate_lst, start=1)}
        return

    def facts_to_sequences(self, facts):
        """
        Transform each fact in facts as a sequence of symbol indexes.
        Only top 'nb_symbols' most frequent symbols will be taken into account.
        Returns a list of sequences.
        :param facts: lists of symbols.
        :return: list of individual sequences of indexes
        """
        return [indices for indices in self.facts_to_sequences_generator(facts)]

    def facts_to_sequences_generator(self, facts):
        """
        Transform each fact in facts as a pair (predicate_idx, argument_idxs),
        where predicate_idx is the index of the predicate, and argument_idxs is a list
        of indices associated to the arguments of the predicate.
        Yields individual pairs.
        :param facts: lists of facts.
        :return: yields individual (predicate_idx, argument_idxs) pairs.
        """
        for fact in facts:
            predicate_idx = self.predicate_index.get(fact.predicate_name)
            argument_idxs = [self.entity_index.get(arg) for arg in fact.argument_names]
            yield (predicate_idx, argument_idxs)

