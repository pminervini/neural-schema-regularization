# -*- coding: utf-8 -*-

from hyper.preprocessing import kb

import logging
import sys


def read_lines(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    return lines


def main(argv):
    logging.info("Loading data...")

    training_lines = read_lines('data/fb15k/freebase_mtr100_mte100-train.txt')

    facts = []
    for line in training_lines:
        subject, predicate, object = line.split()
        facts += [kb.Fact(predicate_name=predicate, argument_names=[subject, object])]

    parser = kb.KnowledgeBaseParser(facts)

    training_pairs = parser.facts_to_sequences(facts)

    for training_pair in training_pairs:
        logging.info('Triple: %s' % str(training_pair))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
