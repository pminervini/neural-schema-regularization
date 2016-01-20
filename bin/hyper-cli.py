#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.constraints import unitnorm
from keras.layers.core import LambdaMerge
from keras.models import make_batches

from keras import backend as K

import hyper.layers.core
from hyper.preprocessing import knowledgebase
from hyper.learning import samples
from hyper import optimizers

import sys
import logging
import argparse

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2015'


def experiment(train_sequences, nb_entities, nb_predicates, seed=1,
               entity_embedding_size=100, predicate_embedding_size=100,
               model_name='ScalE', similarity_name='DOT', nb_epochs=1000, batch_size=128,
               optimizer_name='adagrad', lr=0.1, momentum=0.9, decay=.0, nesterov=False,
               epsilon=1e-6, rho=0.9, beta_1=0.9, beta_2=0.999):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed=seed)

    predicate_encoder = Sequential()
    entity_encoder = Sequential()

    predicate_embedding_layer = Embedding(input_dim=nb_predicates, output_dim=predicate_embedding_size, input_length=1)
    predicate_encoder.add(predicate_embedding_layer)

    entity_embedding_layer = Embedding(input_dim=nb_entities, output_dim=entity_embedding_size,
                                       input_length=2, W_constraint=unitnorm())
    entity_encoder.add(entity_embedding_layer)

    model = Sequential()

    core = sys.modules['hyper.layers.core']
    setattr(core, 'similarity function', similarity_name)
    setattr(core, 'merge function', model_name)

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

    margin = 1

    def margin_based_loss(y_true, y_pred):
        pos = y_pred[0::3]
        neg_subj = y_pred[1::3]
        neg_obj = y_pred[2::3]

        diff_subj = K.clip((neg_subj - pos + margin), 0, np.inf).sum(axis=1, keepdims=True)
        diff_obj = K.clip((neg_obj - pos + margin), 0, np.inf).sum(axis=1, keepdims=True)

        y_true = y_true[0::3]

        return K.abs(diff_subj - y_true).sum() + K.abs(diff_obj - y_true).sum()

    optimizer = optimizers.make_optimizer(optimizer_name, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov,
                                          epsilon=epsilon, rho=rho, beta_1=beta_1, beta_2=beta_2)
    model.compile(loss=margin_based_loss, optimizer=optimizer)

    Xr = np.array([[rel_idx] for (rel_idx, _) in train_sequences])
    Xe = np.array([ent_idxs for (_, ent_idxs) in train_sequences])

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

        nXe_subj_shuffled = np.copy(Xe_shuffled)
        nXe_subj_shuffled[:, 0] = negative_subjects

        nXe_obj_shuffled = np.copy(Xe_shuffled)
        nXe_obj_shuffled[:, 1] = negative_objects

        batches = make_batches(nb_samples, batch_size)

        cumulative_loss = 0.0

        # Iterate over batches of (positive) training examples
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            Xr_batch = Xr_shuffled[batch_start:batch_end]

            Xe_batch = Xe_shuffled[batch_start:batch_end]
            nXe_subj_batch = nXe_subj_shuffled[batch_start:batch_end]
            nXe_obj_batch = nXe_obj_shuffled[batch_start:batch_end]

            assert Xr_batch.shape[0] == Xe_batch.shape[0]
            assert Xr_batch.shape[0] == nXe_subj_batch.shape[0]

            sXr_batch = np.empty((Xr_batch.shape[0] * 3, Xr_batch.shape[1]))
            sXr_batch[0::3] = Xr_batch
            sXr_batch[1::3] = Xr_batch
            sXr_batch[2::3] = Xr_batch

            sXe_batch = np.empty((Xe_batch.shape[0] * 3, Xe_batch.shape[1]))
            sXe_batch[0::3] = Xe_batch
            sXe_batch[1::3] = nXe_subj_batch
            sXe_batch[2::3] = nXe_obj_batch

            y_batch = np.zeros(sXe_batch.shape[0])

            hist = model.fit([sXr_batch, sXe_batch], y_batch, nb_epoch=1, batch_size=sXe_batch.shape[0], verbose=0)
            cumulative_loss += hist.history['loss'][0]

        logging.info('Cumulative loss: %s' % cumulative_loss)

    return


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Latent Factor Models for Knowledge Hypergraphs', formatter_class=formatter)

    argparser.add_argument('--train', required=True, type=argparse.FileType('r'))
    argparser.add_argument('--seed', action='store', type=int, default=1, help='Seed for the PRNG')

    argparser.add_argument('--entity-embedding-size', action='store', type=int, default=100,
                           help='Size of entity embeddings')
    argparser.add_argument('--predicate-embedding-size', action='store', type=int, default=100,
                           help='Size of predicate embeddings')

    argparser.add_argument('--model', action='store', type=str, default=None, help='Name of the model to use')
    argparser.add_argument('--similarity', action='store', type=str, default=None,
                           help='Name of the similarity function to use (if distance-based model)')
    argparser.add_argument('--epochs', action='store', type=int, default=10, help='Number of training epochs')
    argparser.add_argument('--batch-size', action='store', type=int, default=128, help='Batch size')

    argparser.add_argument('--optimizer', action='store', type=str, default='adagrad',
                           help='Optimization algorithm to use - sgd, adagrad, adadelta, rmsprop, adam, adamax')
    argparser.add_argument('--lr', action='store', type=float, default=0.01, help='Learning rate')
    argparser.add_argument('--momentum', action='store', type=float, default=0.0,
                           help='Momentum parameter of the SGD optimizer')
    argparser.add_argument('--decay', action='store', type=float, default=0.0,
                           help='Decay parameter of the SGD optimizer')
    argparser.add_argument('--nesterov', action='store_true', help='Applies Nesterov momentum to the SGD optimizer')
    argparser.add_argument('--epsilon', action='store', type=float, default=1e-06,
                           help='Epsilon parameter of the adagrad, adadelta, rmsprop, adam and adamax optimizers')
    argparser.add_argument('--rho', action='store', type=float, default=0.95,
                           help='Rho parameter of the adadelta and rmsprop optimizers')
    argparser.add_argument('--beta1', action='store', type=float, default=0.9,
                           help='Beta1 parameter for the adam and adamax optimizers')
    argparser.add_argument('--beta2', action='store', type=float, default=0.999,
                           help='Beta2 parameter for the adam and adamax optimizers')

    args = argparser.parse_args(argv)

    train_facts = []
    for line in args.train:
        subj, pred, obj = line.split()
        train_facts += [knowledgebase.Fact(predicate_name=pred, argument_names=[subj, obj])]
    parser = knowledgebase.KnowledgeBaseParser(train_facts)
    nb_entities = len(parser.entity_vocabulary) + 1
    nb_predicates = len(parser.predicate_vocabulary) + 1

    seed = args.seed

    entity_embedding_size = args.entity_embedding_size
    predicate_embedding_size = args.predicate_embedding_size

    model_name = args.model
    similarity_name = args.similarity
    nb_epochs = args.epochs
    batch_size = args.batch_size

    optimizer_name = args.optimizer
    lr = args.lr
    momentum = args.momentum
    decay = args.decay
    nesterov = args.nesterov
    epsilon = args.epsilon
    rho = args.rho
    beta_1 = args.beta1
    beta_2 = args.beta2

    train_sequences = parser.facts_to_sequences(train_facts)

    experiment(train_sequences, nb_entities, nb_predicates, seed=seed,
               entity_embedding_size=entity_embedding_size, predicate_embedding_size=predicate_embedding_size,
               model_name=model_name, similarity_name=similarity_name, nb_epochs=nb_epochs, batch_size=batch_size,
               optimizer_name=optimizer_name, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov,
               epsilon=epsilon, rho=rho, beta_1=beta_1, beta_2=beta_2)
    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
