#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from hyper.parsing import knowledgebase
from hyper import optimizers
from hyper.regularizers import GroupRegularizer, TranslationRuleRegularizer, ScalingRuleRegularizer

from hyper.pathranking.api import PathRankingClient
from hyper.evaluation import metrics

import hyper.learning.core as learning

import sys
import gzip
import logging
import argparse

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2016'


def read_triples(path):
    triples = None
    if path is not None:
        logging.info('Acquiring %s ..' % path)
        my_open = gzip.open if path.endswith('.gz') else open
        with my_open(path, 'rt') as f:
            lines = f.readlines()
        triples = [(s.strip(), p.strip(), o.strip()) for [s, p, o] in [l.split() for l in lines]]
    return triples


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
    for n in [1, 3, 5, 10]:
        metrics.ranking_summary(res, tag=tag, n=n)

    return res


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Latent Factor Models for Knowledge Hypergraphs', formatter_class=formatter)

    argparser.add_argument('--train', required=True, action='store', type=str, default=None)
    argparser.add_argument('--validation', required=False, action='store', type=str, default=None)
    argparser.add_argument('--test', required=False, action='store', type=str, default=None)

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
    argparser.add_argument('--rules-threshold', action='store', type=float, default=None,
                           help='Only show the rules with a score above the given threshold')
    argparser.add_argument('--rules-max-length', action='store', type=int, default=None,
                           help='Maximum (body) length for the considered rules')

    argparser.add_argument('--sample-facts', action='store', type=float, default=None,
                           help='Fraction of (randomly sampled) facts to use during training')

    argparser.add_argument('--rules-lambda', action='store', type=float, default=None,
                           help='Weight of the Rules-related regularization term')

    argparser.add_argument('--model', action='store', type=str, default=None,
                           help='Name of the model to use')
    argparser.add_argument('--similarity', action='store', type=str, default=None,
                           help='Name of the similarity function to use (if distance-based model)')
    argparser.add_argument('--epochs', action='store', type=int, default=10,
                           help='Number of training epochs')
    argparser.add_argument('--batch-size', action='store', type=int, default=128,
                           help='Batch size')
    argparser.add_argument('--batches', action='store', type=int, default=None,
                           help='Number of batches')
    argparser.add_argument('--margin', action='store', type=float, default=1.0,
                           help='Margin to use in the hinge loss')

    argparser.add_argument('--loss', action='store', type=str, default='hinge',
                           help='Loss function to be used (e.g. hinge, logistic)')
    argparser.add_argument('--negatives', action='store', type=str, default='corrupt',
                           help='Method for generating the negative examples (e.g. corrupt, lcwa, schema, bernoulli)')

    argparser.add_argument('--optimizer', action='store', type=str, default='adagrad',
                           help='Optimization algorithm to use - sgd, adagrad, adadelta, rmsprop, adam, adamax')
    argparser.add_argument('--lr', '--optimizer-lr', action='store', type=float, default=0.01, help='Learning rate')
    argparser.add_argument('--optimizer-momentum', action='store', type=float, default=0.0,
                           help='Momentum parameter of the SGD optimizer')
    argparser.add_argument('--optimizer-decay', action='store', type=float, default=0.0,
                           help='Decay parameter of the SGD optimizer')
    argparser.add_argument('--optimizer-nesterov', action='store_true',
                           help='Applies Nesterov momentum to the SGD optimizer')
    argparser.add_argument('--optimizer-epsilon', action='store', type=float, default=1e-06,
                           help='Epsilon parameter of the adagrad, adadelta, rmsprop, adam and adamax optimizers')
    argparser.add_argument('--optimizer-rho', action='store', type=float, default=0.95,
                           help='Rho parameter of the adadelta and rmsprop optimizers')
    argparser.add_argument('--optimizer-beta1', action='store', type=float, default=0.9,
                           help='Beta1 parameter for the adam and adamax optimizers')
    argparser.add_argument('--optimizer-beta2', action='store', type=float, default=0.999,
                           help='Beta2 parameter for the adam and adamax optimizers')

    argparser.add_argument('--tensorflow', '--tf', action='store_true', help='Uses TensorFlow')

    argparser.add_argument('--save', action='store', type=str, default=None,
                           help='Where to save the trained model')

    args = argparser.parse_args(argv)

    if args.tensorflow is True:
        from keras import backend as K
        import tensorflow as tf

        sess = tf.Session()
        K.set_session(sess)

    def to_fact(s, p, o):
        return knowledgebase.Fact(predicate_name=p, argument_names=[s, o])

    train_facts = [to_fact(s, p, o) for s, p, o in read_triples(args.train)]
    validation_facts = [to_fact(s, p, o) for s, p, o in read_triples(args.validation)]\
        if args.validation is not None else []
    test_facts = [to_fact(s, p, o) for s, p, o in read_triples(args.test)]\
        if args.test is not None else []

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

    loss_name = args.loss
    negatives_name = args.negatives

    # Dropout-related parameters
    dropout_entity_embeddings = args.dropout_entity_embeddings
    dropout_predicate_embeddings = args.dropout_predicate_embeddings

    # Rules-related parameters
    rules = args.rules
    rules_top_k = args.rules_top_k
    rules_threshold = args.rules_threshold
    rules_max_length = args.rules_max_length

    sample_facts = args.sample_facts

    rules_lambda = args.rules_lambda

    rule_regularizer = None

    if rules is not None and rules_lambda is not None and rules_lambda > .0:
        path_ranking_client = PathRankingClient(url_or_path=rules)
        pfw_triples = path_ranking_client.request(None, threshold=.0, top_k=rules_top_k)
        model_to_regularizer = dict(TransE=TranslationRuleRegularizer, ScalE=ScalingRuleRegularizer)

        rule_regularizers = []
        for rule_predicate, rule_feature, rule_weight in pfw_triples:
            if rules_threshold is None or rule_weight >= rules_threshold:
                if rules_max_length is None or len(rule_feature.hops) <= rules_max_length:
                    head = parser.predicate_index[rule_predicate]
                    tail = [(parser.predicate_index[hop.predicate], hop.is_inverse) for hop in rule_feature.hops]

                    logging.info('[Rules] Adding Head: %s, Tail: %s' % (str(head), str(tail)))

                    if model_name not in model_to_regularizer:
                        raise ValueError('Rule-based regularizers unsupported for the model: %s' % model_name)

                    Regularizer = model_to_regularizer[model_name]
                    rule_regularizers += [Regularizer(head, tail, l=rules_lambda)]

        rule_regularizer = GroupRegularizer(regularizers=rule_regularizers) if rule_regularizers else None

    if sample_facts is not None:
        nb_train_facts = len(train_facts)
        sample_size = int(round(sample_facts * nb_train_facts))

        random_state = np.random.RandomState(seed)
        sample_indices = random_state.choice(nb_train_facts, sample_size, replace=False)
        train_facts_sample = [train_facts[i] for i in sample_indices]

        train_facts = train_facts_sample

    optimizer_name = args.optimizer
    optimizer_lr = args.lr
    optimizer_momentum = args.optimizer_momentum
    optimizer_decay = args.optimizer_decay
    optimizer_nesterov = args.optimizer_nesterov
    optimizer_epsilon = args.optimizer_epsilon
    optimizer_rho = args.optimizer_rho
    optimizer_beta_1 = args.optimizer_beta1
    optimizer_beta_2 = args.optimizer_beta2

    optimizer = optimizers.make_optimizer(optimizer_name,
                                          lr=optimizer_lr, momentum=optimizer_momentum,
                                          decay=optimizer_decay, nesterov=optimizer_nesterov,
                                          epsilon=optimizer_epsilon, rho=optimizer_rho,
                                          beta_1=optimizer_beta_1, beta_2=optimizer_beta_2)

    train_sequences = parser.facts_to_sequences(train_facts)

    model = learning.pairwise_training(train_sequences, nb_entities, nb_predicates, seed=seed,
                                       entity_embedding_size=entity_embedding_size,
                                       predicate_embedding_size=predicate_embedding_size,

                                       dropout_entity_embeddings=dropout_entity_embeddings,
                                       dropout_predicate_embeddings=dropout_predicate_embeddings,

                                       model_name=model_name, similarity_name=similarity_name,
                                       nb_epochs=nb_epochs, batch_size=batch_size, nb_batches=nb_batches, margin=margin,
                                       loss_name=loss_name, negatives_name=negatives_name,

                                       optimizer=optimizer, rule_regularizer=rule_regularizer)

    if args.save is not None:
        pass

    validation_sequences = parser.facts_to_sequences(validation_facts)
    test_sequences = parser.facts_to_sequences(test_facts)

    true_triples = np.array([[s, p, o] for (p, [s, o]) in train_sequences + validation_sequences + test_sequences])

    if len(validation_sequences) > 0:
        evaluate_model(model, validation_sequences, nb_entities, tag='validation raw')
        evaluate_model(model, validation_sequences, nb_entities, true_triples=true_triples, tag='validation filtered')

    if len(test_sequences) > 0:
        evaluate_model(model, test_sequences, nb_entities, tag='test raw')
        evaluate_model(model, test_sequences, nb_entities, true_triples=true_triples, tag='test filtered')

    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.setrecursionlimit(65536)
    main(sys.argv[1:])
