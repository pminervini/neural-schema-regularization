#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.constraints import Constraint
from keras.layers.core import LambdaMerge
from keras.models import make_batches

from keras import backend as K

import hyper.layers.core
from hyper.preprocessing import knowledgebase
from hyper.learning import samples
from hyper import optimizers

from hyper.evaluation import metrics

import sys
import logging
import inspect
import argparse

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2015'


class FixedNorm(Constraint):
    def __init__(self, m=1.):
        self.m = m

    def __call__(self, p):
        p = K.transpose(p)
        unit_norm = p / (K.sqrt(K.sum(K.square(p), axis=0)) + 1e-7)
        unit_norm = K.transpose(unit_norm)
        return unit_norm * self.m

    def get_config(self):
        return {'name': self.__class__.__name__, 'm': self.m}


def train_model(train_sequences, nb_entities, nb_predicates, seed=1,
                entity_embedding_size=100, predicate_embedding_size=100,
                model_name='TransE', similarity_name='L1', nb_epochs=1000, batch_size=128, nb_batches=None, margin=1.0,
                optimizer_name='adagrad', lr=0.1, momentum=0.9, decay=.0, nesterov=False,
                epsilon=1e-6, rho=0.9, beta_1=0.9, beta_2=0.999):

    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    logging.info('Experiment: %s' % {arg: values[arg] for arg in args if len(str(values[arg])) < 32})

    np.random.seed(seed)
    random_state = np.random.RandomState(seed=seed)

    predicate_encoder = Sequential()
    entity_encoder = Sequential()

    predicate_embedding_layer = Embedding(input_dim=nb_predicates + 1, output_dim=predicate_embedding_size,
                                          input_length=None, init='glorot_uniform')
    predicate_encoder.add(predicate_embedding_layer)

    entity_embedding_layer = Embedding(input_dim=nb_entities + 1, output_dim=entity_embedding_size,
                                       input_length=None, init='glorot_uniform', W_constraint=FixedNorm())
    entity_encoder.add(entity_embedding_layer)

    model = Sequential()

    core = sys.modules['hyper.layers.core']
    setattr(core, 'similarity function', similarity_name)
    setattr(core, 'merge function', model_name)

    def f(args):
        import sys
        import hyper.similarities as similarities
        import hyper.layers.binary.merge_functions as merge_functions

        f_core = sys.modules['hyper.layers.core']
        similarity_function_name = getattr(f_core, 'similarity function')
        merge_function_name = getattr(f_core, 'merge function')

        similarity_function = similarities.get_function(similarity_function_name)
        merge_function = merge_functions.get_function(merge_function_name)

        return merge_function(args, similarity=similarity_function)

    merge_layer = LambdaMerge([predicate_encoder, entity_encoder], function=f)
    model.add(merge_layer)

    def margin_based_loss(y_true, y_pred):
        pos, neg_subj, neg_obj = y_pred[0::3], y_pred[1::3], y_pred[2::3]

        out_subj = (margin + neg_subj - pos)
        diff_subj = out_subj * K.cast(out_subj >= 0., K.floatx()).sum(axis=1, keepdims=True)

        out_obj = (margin + neg_obj - pos)
        diff_obj = out_obj * K.cast(out_obj >= 0., K.floatx()).sum(axis=1, keepdims=True)

        target = y_true[0::3]

        return diff_subj.sum() + diff_obj.sum() + target.sum()

    optimizer = optimizers.make_optimizer(optimizer_name, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov,
                                          epsilon=epsilon, rho=rho, beta_1=beta_1, beta_2=beta_2)
    model.compile(loss=margin_based_loss, optimizer=optimizer)

    Xr = np.array([[rel_idx] for (rel_idx, _) in train_sequences])
    Xe = np.array([ent_idxs for (_, ent_idxs) in train_sequences])

    print(Xr.shape, Xr.min(), Xr.max())
    print(Xe.shape, Xe.min(), Xe.max())

    nb_samples = Xr.shape[0]

    if nb_batches is not None:
        batch_size = math.ceil(nb_samples / nb_batches)
        logging.info("Samples: %d, no. batches: %d -> batch size: %d" % (nb_samples, nb_batches, batch_size))

    # Random index generator for sampling negative examples
    random_index_generator = samples.GlorotRandomIndexGenerator(random_state=random_state)

    # Creating negative indices..
    candidate_negative_indices = np.arange(1, nb_entities + 1)

    for epoch_no in range(1, nb_epochs + 1):
        logging.info('Epoch no. %d of %d (samples: %d)' % (epoch_no, nb_epochs, nb_samples))

        # Shuffling training (positive) triples..
        order = random_state.permutation(nb_samples)

        Xr_shuffled, Xe_shuffled = Xr[order, :], Xe[order, :]

        nXe_subj_shuffled = np.copy(Xe_shuffled)

        negative_subjects = random_index_generator.generate(nb_samples, candidate_negative_indices)
        nXe_subj_shuffled[:, 0] = negative_subjects

        nXe_obj_shuffled = np.copy(Xe_shuffled)

        negative_objects = random_index_generator.generate(nb_samples, candidate_negative_indices)
        nXe_obj_shuffled[:, 1] = negative_objects

        batches = make_batches(nb_samples, batch_size)

        losses = []

        # Iterate over batches of (positive) training examples
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            logging.info('Batch no. %d of %d (%d:%d), size %d'
                         % (batch_index, len(batches), batch_start, batch_end, batch_end - batch_start))

            Xr_batch = Xr_shuffled[batch_start:batch_end, :]

            Xe_batch = Xe_shuffled[batch_start:batch_end, :]
            nXe_subj_batch = nXe_subj_shuffled[batch_start:batch_end, :]
            nXe_obj_batch = nXe_obj_shuffled[batch_start:batch_end, :]

            sXr_batch = np.empty((Xr_batch.shape[0] * 3, Xr_batch.shape[1]))
            sXr_batch[0::3, :] = Xr_batch
            sXr_batch[1::3, :] = Xr_batch
            sXr_batch[2::3, :] = Xr_batch

            sXe_batch = np.empty((Xe_batch.shape[0] * 3, Xe_batch.shape[1]))
            sXe_batch[0::3, :] = Xe_batch
            sXe_batch[1::3, :] = nXe_subj_batch
            sXe_batch[2::3, :] = nXe_obj_batch

            y_batch = np.zeros(sXe_batch.shape[0])

            hist = model.fit([sXr_batch, sXe_batch], y_batch, nb_epoch=1, batch_size=sXe_batch.shape[0],
                             shuffle=False, verbose=0)

            loss = hist.history['loss'][0]
            losses += [loss / sXr_batch.shape[0]]

        logging.info('Loss: %s +/- %s' % (round(np.mean(losses), 4), round(np.std(losses), 4)))

    return model


def evaluate_model(model, validation_sequences, nb_entities):

    def scoring_function(args):
        Xr, Xe = args[0], args[1]
        y = model.predict([Xr, Xe], batch_size=Xr.shape[0])
        return y

    validation_triples = [(s, p, o) for (p, [s, o]) in validation_sequences]
    res = metrics.ranking_score(scoring_function, validation_triples, nb_entities, nb_entities)
    metrics.ranking_summary(res)


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Latent Factor Models for Knowledge Hypergraphs', formatter_class=formatter)

    argparser.add_argument('--train', required=True, type=argparse.FileType('r'))
    argparser.add_argument('--validation', required=False, type=argparse.FileType('r'))

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
    argparser.add_argument('--batches', action='store', type=int, default=None, help='Number of batches')
    argparser.add_argument('--margin', action='store', type=float, default=1.0, help='Margin to use in the hinge loss')

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

    argparser.add_argument('--save', action='store', type=str, default=None,
                           help='Where to save the trained model')

    args = argparser.parse_args(argv)

    train_facts = []
    for line in args.train:
        subj, pred, obj = line.split()
        train_facts += [knowledgebase.Fact(predicate_name=pred, argument_names=[subj, obj])]

    validation_facts = []
    if args.validation is not None:
        for line in args.validation:
            sub, pred, obj = line.split()
            validation_facts += [knowledgebase.Fact(predicate_name=pred, argument_names=[subj, obj])]

    parser = knowledgebase.KnowledgeBaseParser(train_facts + validation_facts)

    nb_entities = len(parser.entity_vocabulary)
    nb_predicates = len(parser.predicate_vocabulary)

    seed = args.seed

    entity_embedding_size = args.entity_embedding_size
    predicate_embedding_size = args.predicate_embedding_size

    model_name = args.model
    similarity_name = args.similarity
    nb_epochs = args.epochs
    batch_size = args.batch_size
    nb_batches = args.batches
    margin = args.margin

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

    model = train_model(train_sequences, nb_entities, nb_predicates, seed=seed,
                        entity_embedding_size=entity_embedding_size, predicate_embedding_size=predicate_embedding_size,
                        model_name=model_name, similarity_name=similarity_name,
                        nb_epochs=nb_epochs, batch_size=batch_size, nb_batches=nb_batches, margin=margin,
                        optimizer_name=optimizer_name, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov,
                        epsilon=epsilon, rho=rho, beta_1=beta_1, beta_2=beta_2)

    if args.save is not None:
        pass

    validation_sequences = parser.facts_to_sequences(validation_facts)
    evaluate_model(model, validation_sequences, nb_entities)

    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
