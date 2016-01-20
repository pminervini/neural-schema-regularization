#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import LambdaMerge
from keras.models import make_batches

from keras import backend as K

from hyper.layers.core import Arguments
from hyper.preprocessing import knowledgebase
from hyper.learning import samples

import sys
import logging
import argparse

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2015'


def experiment(train_sequences, nb_entities, nb_predicates,
               entity_embedding_size=100, predicate_embedding_size=100,
               nb_epochs=100, batch_size=128, seed=1):

    np.random.seed(seed)
    random_state = np.random.RandomState(seed=seed)

    predicate_encoder = Sequential()
    entity_encoder = Sequential()

    predicate_embedding_layer = Embedding(input_dim=nb_predicates, output_dim=predicate_embedding_size, input_length=1)
    predicate_encoder.add(predicate_embedding_layer)

    entity_embedding_layer = Embedding(input_dim=nb_entities, output_dim=entity_embedding_size, input_length=None)
    entity_encoder.add(entity_embedding_layer)

    model = Sequential()

    core = sys.modules['hyper.layers.core']
    setattr(core, 'similarity function', 'L2')
    setattr(core, 'merge function', 'ScalE')

    def f(args):
        import sys
        import hyper.similarities as similarities
        import hyper.layers.binary.merge_functions as merge_functions

        core = sys.modules['hyper.layers.core']
        similarity_function_name = getattr(core, 'similarity function')
        merge_function_name = getattr(core, 'merge function')

        similarity_function = similarities.get_function(similarity_function_name)
        merge_function = merge_functions.get_function(merge_function_name)

        return merge_function(args, similarity=similarity_function)

    merge_layer = LambdaMerge([predicate_encoder, entity_encoder], function=f)
    model.add(merge_layer)

    def margin_based_loss(y_true, y_pred):
        pos = y_pred[0::2]
        neg = y_pred[1::2]
        diff = (K.clip((neg - pos + 1), 0, np.inf)).sum(axis=1, keepdims=True)
        y_true = y_true[0::2]
        return diff.mean()

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    Xr = np.array([[rel_idx] for (rel_idx, _) in train_sequences])
    Xe = np.array([ent_idxs for (_, ent_idxs) in train_sequences])

    y = model.predict([Xr, Xe], batch_size=1)

    nb_samples = Xr.shape[0]
    assert Xr.shape[0] == Xe.shape[0]

    # Random index generator for sampling negative examples
    random_index_generator = samples.UniformRandomIndexGenerator(random_state=random_state)

    for epoch_no in range(1, nb_epochs + 1):
        logging.info('Epoch no. %d of %d' % (epoch_no, nb_epochs))

        # Shuffling training (positive) triples..
        order = random_state.permutation(nb_samples)

        Xr_shuffled = Xr[order, :]
        Xe_shuffled = Xe[order, :]

        assert Xr_shuffled.shape[0] == Xe_shuffled.shape[0]

        # Creating negative examples..
        candidate_indices = np.arange(1, nb_entities)

        negative_subjects = random_index_generator.generate(nb_samples, candidate_indices)
        negative_objects = random_index_generator.generate(nb_samples, candidate_indices)

        # Negative examples (with different subjects and objects)
        # TODO: I should corrupt only subjects first, and then only objects (LCWA)
        nXe_shuffled = np.copy(Xe_shuffled)
        #nXe_shuffled[:, 0] = negative_subjects
        nXe_shuffled[:, 1] = negative_objects

        batches = make_batches(nb_samples, batch_size)

        # Iterate over batches of (positive) training examples
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            Xr_batch = Xr_shuffled[batch_start:batch_end]

            Xe_batch = Xe_shuffled[batch_start:batch_end]
            nXe_batch = nXe_shuffled[batch_start:batch_end]

            assert Xr_batch.shape[0] == Xe_batch.shape[0]
            assert Xr_batch.shape[0] == nXe_batch.shape[0]

            sXr_batch = np.empty((Xr_batch.shape[0] * 2, Xr_batch.shape[1]))
            sXr_batch[0::2] = Xr_batch
            sXr_batch[1::2] = Xr_batch

            sXe_batch = np.empty((Xe_batch.shape[0] * 2, Xe_batch.shape[1]))
            sXe_batch[0::2] = Xe_batch
            sXe_batch[1::2] = nXe_batch

            y_batch = np.zeros(Xe_batch.shape[0] * 2)

            model.fit([sXr_batch, sXe_batch], y_batch)

    return


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Latent Factor Models for Knowledge Hypergraphs', formatter_class=formatter)

    argparser.add_argument('--train', required=True, type=argparse.FileType('r'))
    argparser.add_argument('--epochs', action='store', type=int, default=10, help='Number of training epochs')
    argparser.add_argument('--batch-size', action='store', type=int, default=128, help='Batch size')
    argparser.add_argument('--seed', action='store', type=int, default=1, help='Seed for the PRNG')

    argparser.add_argument('--entity-embedding-size', action='store', type=int, default=100,
                           help='Size of entity embeddings')
    argparser.add_argument('--predicate-embedding-size', action='store', type=int, default=100,
                           help='Size of predicate embeddings')

    args = argparser.parse_args(argv)

    train_facts = []
    for line in args.train:
        subj, pred, obj = line.split()
        train_facts += [knowledgebase.Fact(predicate_name=pred, argument_names=[subj, obj])]

    parser = knowledgebase.KnowledgeBaseParser(train_facts)

    epochs = args.epochs
    batch_size = args.batch_size
    seed = args.seed

    nb_entities = len(parser.entity_vocabulary) + 1
    nb_predicates = len(parser.predicate_vocabulary) + 1

    entity_embedding_size = args.entity_embedding_size
    predicate_embedding_size = args.predicate_embedding_size

    train_sequences = parser.facts_to_sequences(train_facts)

    experiment(train_sequences, nb_entities, nb_predicates,
               entity_embedding_size=entity_embedding_size,
               predicate_embedding_size=predicate_embedding_size,
               nb_epochs=epochs, batch_size=batch_size,
               seed=seed)
    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
