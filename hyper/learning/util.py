# -*- coding: utf-8 -*-

from enum import Enum
import logging


class PredicateType(Enum):
    one_to_one = 1
    one_to_many = 2
    many_to_one = 3
    many_to_many = 4

    def __str__(self):
        return self.name


def find_predicate_types(Xr, Xe):
    logging.info('Recognizing the type of each predicate (1-to-1, 1-to-M, M-to-1, M-to-M) ..')

    # entities = set([s for [s, _] in Xe] + [o for [_, o] in Xe])
    predicates = set([p for [p] in Xr])

    ps_count = dict()  # {(p, s): 0 for p in predicates for s in entities}
    po_count = dict()  # {(p, o): 0 for p in predicates for o in entities}

    for [p], [s, o] in zip(Xr, Xe):
        if (p, s) not in ps_count:
            ps_count[(p, s)] = 0
        if (p, o) not in po_count:
            po_count[(p, o)] = 0

        ps_count[(p, s)] += 1
        po_count[(p, o)] += 1

    predicate2type = dict()

    for p in predicates:
        at_most_one_o_per_s = True
        at_most_one_s_per_o = True

        for (_p, _), count in ps_count.items():
            if p == _p and count > 1:
                at_most_one_o_per_s = False

        for (_p, _), count in po_count.items():
            if p == _p and count > 1:
                at_most_one_s_per_o = False

        predicate_type = None
        if at_most_one_o_per_s is True and at_most_one_s_per_o is True:
            predicate_type = PredicateType.one_to_one
        elif at_most_one_o_per_s is True and at_most_one_s_per_o is False:
            predicate_type = PredicateType.many_to_one
        elif at_most_one_o_per_s is False and at_most_one_s_per_o is True:
            predicate_type = PredicateType.one_to_many
        elif at_most_one_o_per_s is False and at_most_one_s_per_o is False:
            predicate_type = PredicateType.many_to_many

        predicate2type[p] = predicate_type

    return predicate2type
