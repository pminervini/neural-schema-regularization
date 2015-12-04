# -*- coding: utf-8 -*-

import logging

class KnowledgeBaseParser(object):
    def __init__(self, facts, nb_symbols=None):
        self.nb_symbols = nb_symbols
        self.symbol_counts = dict()
        self.symbol_index = dict()
        self._fit_on_facts(facts)

    def _fit_on_facts(self, facts):
        """
        Required before using facts_to_sequences
        :param facts: List or generator of facts.
        :return:
        """
        for fact in facts:
            for sym in fact:
                occurrences = self.symbol_counts[sym] + 1 if sym in self.symbol_counts else 1
                self.symbol_counts[sym] = occurrences

        symbol_count_pairs = list(self.symbol_counts.items())
        symbol_count_pairs.sort(key=lambda x: x[1], reverse=True)

        sorted_symbol_vocabulary = [symbol_count_pair[0] for symbol_count_pair in symbol_count_pairs]

        self.symbol_index = {symbol: index for index, symbol in enumerate(sorted_symbol_vocabulary, start=1)}
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
        Transform each fact in facts as a sequence of symbol indexes.
        Only top 'nb_symbols' most frequent symbols will be taken into account.
        Yields individual sequences.
        :param facts: lists of symbols.
        :return: yields individual sequences of indexes.
        """
        for fact in facts:
            indices = []
            for sym in fact:
                index = self.symbol_index.get(sym)
                if index is not None and (self.nb_symbols is None or index <= self.nb_symbols):
                    indices.append(index)
            yield indices

