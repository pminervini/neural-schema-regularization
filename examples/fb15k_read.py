# -*- coding: utf-8 -*-

from hyper.preprocessing import kb

from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, TimeDistributedMerge
from keras.models import Graph

import logging
import sys


def read_lines(fname):
    with open(fname) as f:
        content = f.readlines()
    return content


def main(argv):
    print("Loading data...")

    training_lines = read_lines('data/fb15k/freebase_mtr100_mte100-train.txt')

    facts = list()
    for line in training_lines:
        subject, predicate, object = line.split()
        facts.append(kb.Fact(predicate_name=predicate, argument_names=[subject, object]))

    parser = kb.KnowledgeBaseParser(facts)

    training_pairs = parser.facts_to_sequences(facts)

    for training_pair in training_pairs:
        print(training_pair)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
