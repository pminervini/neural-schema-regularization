#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, LambdaMerge
from keras.models import make_batches

from keras import backend as K

import hyper.layers.core
from hyper.preprocessing import knowledgebase
from hyper.learning import samples
from hyper import optimizers, constraints, regularizers

from hyper.pathranking.api import PathRankingClient

from hyper.evaluation import metrics

import sys
import logging
import inspect
import argparse

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2016'


def train_model(train_sequences, nb_entities, nb_predicates, seed=1,
                entity_embedding_size=100, predicate_embedding_size=100,

                dropout_entity_embeddings=None, dropout_predicate_embeddings=None,

                model_name='TransE', similarity_name='L1', nb_epochs=1000, batch_size=128, nb_batches=None, margin=1.0,
                optimizer_name='adagrad', lr=0.1, momentum=0.9, decay=.0, nesterov=False,
                epsilon=1e-6, rho=0.9, beta_1=0.9, beta_2=0.999,
                rule_regularizer=None):

    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    logging.debug('Experiment: %s' % {arg: values[arg] for arg in args if len(str(values[arg])) < 32})

    np.random.seed(seed)
    random_state = np.random.RandomState(seed=seed)

    predicate_encoder = Sequential()
    entity_encoder = Sequential()

    predicate_embedding_layer = Embedding(input_dim=nb_predicates + 1, output_dim=predicate_embedding_size,
                                          input_length=None, init='glorot_uniform', W_regularizer=rule_regularizer)
    predicate_encoder.add(predicate_embedding_layer)

    if dropout_predicate_embeddings is not None and dropout_predicate_embeddings > .0:
        predicate_encoder.add(Dropout(dropout_predicate_embeddings))

    entity_embedding_layer = Embedding(input_dim=nb_entities + 1, output_dim=entity_embedding_size,
                                       input_length=None, init='glorot_uniform',
                                       W_constraint=constraints.NormConstraint(norm=1.))
    entity_encoder.add(entity_embedding_layer)

    if dropout_entity_embeddings is not None and dropout_entity_embeddings > .0:
        entity_encoder.add(Dropout(dropout_entity_embeddings))

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
        diff_subj = (out_subj * K.cast(out_subj >= 0., K.floatx())).sum(axis=1, keepdims=True)

        out_obj = (margin + neg_obj - pos)
        diff_obj = (out_obj * K.cast(out_obj >= 0., K.floatx())).sum(axis=1, keepdims=True)

        target = y_true[0::3]

        return diff_subj.sum() + diff_obj.sum() + target.sum()

    optimizer = optimizers.make_optimizer(optimizer_name, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov,
                                          epsilon=epsilon, rho=rho, beta_1=beta_1, beta_2=beta_2)
    model.compile(loss=margin_based_loss, optimizer=optimizer)

    Xr = np.array([[rel_idx] for (rel_idx, _) in train_sequences])
    Xe = np.array([ent_idxs for (_, ent_idxs) in train_sequences])

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

        negative_subjects = random_index_generator.generate(nb_samples, candidate_negative_indices)
        Xe_neg_subj_shuffled = np.copy(Xe_shuffled)
        Xe_neg_subj_shuffled[:, 0] = negative_subjects

        negative_objects = random_index_generator.generate(nb_samples, candidate_negative_indices)
        Xe_neg_obj_shuffled = np.copy(Xe_shuffled)
        Xe_neg_obj_shuffled[:, 1] = negative_objects

        batches, losses = make_batches(nb_samples, batch_size), []

        # Iterate over batches of (positive) training examples
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            logging.debug('Batch no. %d of %d (%d:%d), size %d'
                          % (batch_index, len(batches), batch_start, batch_end, batch_end - batch_start))

            Xr_batch = Xr_shuffled[batch_start:batch_end, :]

            Xe_batch = Xe_shuffled[batch_start:batch_end, :]
            Xe_neg_subj_batch = Xe_neg_subj_shuffled[batch_start:batch_end, :]
            Xe_neg_obj_batch = Xe_neg_obj_shuffled[batch_start:batch_end, :]

            train_Xr_batch = np.empty((Xr_batch.shape[0] * 3, Xr_batch.shape[1]))
            train_Xr_batch[0::3, :], train_Xr_batch[1::3, :], train_Xr_batch[2::3, :] = Xr_batch, Xr_batch, Xr_batch

            train_Xe_batch = np.empty((Xe_batch.shape[0] * 3, Xe_batch.shape[1]))
            train_Xe_batch[0::3, :], train_Xe_batch[1::3, :], train_Xe_batch[2::3, :] = \
                Xe_batch, Xe_neg_subj_batch, Xe_neg_obj_batch

            y_batch = np.zeros(train_Xr_batch.shape[0])

            hist = model.fit([train_Xr_batch, train_Xe_batch], y_batch, nb_epoch=1, batch_size=train_Xe_batch.shape[0],
                             shuffle=False, verbose=0)

            losses += [hist.history['loss'][0] / train_Xr_batch.shape[0]]

        logging.info('Loss: %s +/- %s' % (round(np.mean(losses), 4), round(np.std(losses), 4)))

    return model


def evaluate_model(model, evaluation_sequences, nb_entities, true_triples=None, tag=None):

    def scoring_function(args):
        Xr, Xe = args[0], args[1]
        y = model.predict([Xr, Xe], batch_size=Xr.shape[0])
        return y[:, 0]

    evaluation_triples = [(s, p, o) for (p, [s, o]) in evaluation_sequences]

    if true_triples is None:
        res = metrics.ranking_score(scoring_function, evaluation_triples, nb_entities, nb_entities)
    else:
        res = metrics.filtered_ranking_score(scoring_function, evaluation_triples,
                                             nb_entities, nb_entities, true_triples)

    metrics.ranking_summary(res, tag=tag)

    return res


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Latent Factor Models for Knowledge Hypergraphs', formatter_class=formatter)

    argparser.add_argument('--train', required=True, type=argparse.FileType('r'))
    argparser.add_argument('--validation', required=False, type=argparse.FileType('r'))
    argparser.add_argument('--test', required=False, type=argparse.FileType('r'))

    argparser.add_argument('--seed', action='store', type=int, default=1, help='Seed for the PRNG')

    argparser.add_argument('--entity-embedding-size', action='store', type=int, default=100,
                           help='Size of entity embeddings')
    argparser.add_argument('--predicate-embedding-size', action='store', type=int, default=100,
                           help='Size of predicate embeddings')

    # Dropout-related arguments
    argparser.add_argument('--dropout-entity-embeddings', action='store', type=float, default=None,
                           help='Dropout after the entity embeddings layer')
    argparser.add_argument('--dropout-predicate-embeddings', action='store', type=float, default=None,
                           help='Dropout after the predicate embeddings layer')

    # Rules-related arguments
    argparser.add_argument('--rules', action='store', type=str, default=None,
                           help='JSON document containing the rules extracted from the KG')
    argparser.add_argument('--rules-top-k', action='store', type=int, default=None,
                           help='Top-k rules to consider during the training process')
    argparser.add_argument('--rules-lambda', action='store', type=float, default=None,
                           help='Weight of the Rules-related regularization term')

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

    def to_fact(line):
        subj, pred, obj = line.split()
        return knowledgebase.Fact(predicate_name=pred, argument_names=[subj, obj])

    train_facts = [to_fact(line) for line in args.train]
    validation_facts = [to_fact(line) for line in args.validation] if args.validation is not None else []
    test_facts = [to_fact(line) for line in args.test] if args.test is not None else []

    parser = knowledgebase.KnowledgeBaseParser(train_facts + validation_facts + test_facts)

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

    # Dropout-related parameters
    dropout_entity_embeddings = args.dropout_entity_embeddings
    dropout_predicate_embeddings = args.dropout_predicate_embeddings

    # Rules-related parameters
    rules = args.rules
    rules_top_k = args.rules_top_k
    rules_lambda = args.rules_lambda

    rule_regularizer = None

    if rules is not None:
        path_ranking_client = PathRankingClient(url_or_path=rules)
        pfw_triples = path_ranking_client.request(None, threshold=.0, top_k=rules_top_k)

        rule_regularizers = []
        for _p, _f, _w in pfw_triples:
            head = parser.predicate_index[_p]
            tail = [(parser.predicate_index[hop.predicate], hop.is_inverse) for hop in _f.hops]

            if model_name == 'TransE':
                rule_regularizers += [regularizers.TranslationRuleRegularizer(head, tail, l=rules_lambda)]
            elif model_name == 'ScalE':
                rule_regularizers += [regularizers.ScalingRuleRegularizer(head, tail, l=rules_lambda)]
            else:
                raise ValueError('Rule-based regularizers unsupported for the model: %s' % model_name)

        rule_regularizer = regularizers.GroupRegularizer(regularizers=rule_regularizers) if rule_regularizers else None

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

                        dropout_entity_embeddings=dropout_entity_embeddings,
                        dropout_predicate_embeddings=dropout_predicate_embeddings,

                        model_name=model_name, similarity_name=similarity_name,
                        nb_epochs=nb_epochs, batch_size=batch_size, nb_batches=nb_batches, margin=margin,
                        optimizer_name=optimizer_name, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov,
                        epsilon=epsilon, rho=rho, beta_1=beta_1, beta_2=beta_2,
                        rule_regularizer=rule_regularizer)

    if args.save is not None:
        pass

    validation_sequences = parser.facts_to_sequences(validation_facts)
    test_sequences = parser.facts_to_sequences(test_facts)

    true_triples = np.array([[s, p, o] for (p, [s, o]) in train_sequences + validation_sequences + test_sequences])

    evaluate_model(model, validation_sequences, nb_entities, tag='validation raw')
    evaluate_model(model, validation_sequences, nb_entities, true_triples=true_triples, tag='validation filtered')

    evaluate_model(model, test_sequences, nb_entities, tag='test raw')
    evaluate_model(model, test_sequences, nb_entities, true_triples=true_triples, tag='test filtered')

    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.setrecursionlimit(65536)
    main(sys.argv[1:])
