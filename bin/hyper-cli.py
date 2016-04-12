#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

from keras.models import Sequential
from keras.layers import Lambda, SimpleRNN, GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Merge, Dropout
from keras.engine.training import make_batches

import hyper.layers.core as core

from hyper.preprocessing import knowledgebase
from hyper.learning import samples, negatives
from hyper import ranking_objectives, optimizers, constraints
from hyper.regularizers import GroupRegularizer, TranslationRuleRegularizer, ScalingRuleRegularizer

import hyper.learning.util as learning_util

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
                loss_name='hinge', negatives_name='corrupt',

                optimizer_name='adagrad',
                optimizer_lr=0.1, optimizer_momentum=0.9, optimizer_decay=.0, optimizer_nesterov=False,
                optimizer_epsilon=1e-6, optimizer_rho=0.9, optimizer_beta_1=0.9, optimizer_beta_2=0.999,
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
                                       W_constraint=constraints.NormConstraint(m=1., axis=1))
    entity_encoder.add(entity_embedding_layer)

    if dropout_entity_embeddings is not None and dropout_entity_embeddings > .0:
        entity_encoder.add(Dropout(dropout_entity_embeddings))

    model = Sequential()

    setattr(core, 'similarity function', similarity_name)
    setattr(core, 'merge function', model_name)

    if model_name in ['TransE', 'ScalE', 'HolE']:
        merge_function = core.latent_distance_binary_merge_function

        merge_layer = Merge([predicate_encoder, entity_encoder], mode=merge_function, output_shape=lambda _: (None, 1))
        model.add(merge_layer)

    elif model_name in ['rTransE', 'rScalE']:
        merge_function = core.latent_distance_nary_merge_function

        merge_layer = Merge([predicate_encoder, entity_encoder], mode=merge_function)
        model.add(merge_layer)

    elif model_name in ['RNN', 'iRNN', 'GRU', 'LSTM']:
        name_to_layer_class = dict(RNN=SimpleRNN, GRU=GRU, LSTM=LSTM)
        if model_name == 'iRNN':
            recurrent_layer = SimpleRNN(output_dim=predicate_embedding_size, return_sequences=False,
                                        init='normal', inner_init='identity', activation='relu')
        elif model_name in name_to_layer_class:
            layer_class = name_to_layer_class[model_name]
            recurrent_layer = layer_class(output_dim=predicate_embedding_size, return_sequences=False)
        else:
            raise ValueError('Unknown recurrent layer: %s' % model_name)
        entity_encoder.add(recurrent_layer)

        merge_function = core.similarity_merge_function

        merge_layer = Merge([predicate_encoder, entity_encoder], mode=merge_function)
        model.add(merge_layer)
    else:
        raise ValueError('Unknown model name: %s' % model_name)

    Xr = np.array([[rel_idx] for (rel_idx, _) in train_sequences])
    Xe = np.array([ent_idxs for (_, ent_idxs) in train_sequences])

    # Let's make the training set unwriteable (immutable), just in case
    Xr.flags.writeable, Xe.flags.writeable = False, False

    nb_samples = Xr.shape[0]

    if nb_batches is not None:
        batch_size = math.ceil(nb_samples / nb_batches)
        logging.info("Samples: %d, no. batches: %d -> batch size: %d" % (nb_samples, nb_batches, batch_size))

    # Random index generator for sampling negative examples
    random_index_generator = samples.GlorotRandomIndexGenerator(random_state=random_state)

    # Creating negative indices..
    candidate_negative_indices = np.arange(1, nb_entities + 1)

    if negatives_name == 'corrupt':
        negative_samples_generator = negatives.CorruptedSamplesGenerator(
            subject_index_generator=random_index_generator, subject_candidate_indices=candidate_negative_indices,
            object_index_generator=random_index_generator, object_candidate_indices=candidate_negative_indices)
    elif negatives_name == 'lcwa':
        negative_samples_generator = negatives.LCWANegativeSamplesGenerator(
            object_index_generator=random_index_generator, object_candidate_indices=candidate_negative_indices)
    elif negatives_name == 'schema':
        predicate2type = learning_util.find_predicate_types(Xr, Xe)
        negative_samples_generator = negatives.SchemaAwareNegativeSamplesGenerator(
            index_generator=random_index_generator, candidate_indices=candidate_negative_indices,
            random_state=random_state, predicate2type=predicate2type)
    elif negatives_name == 'binomial' or negatives_name == 'bernoulli':
        ps_count, po_count = learning_util.predicate_statistics(Xr, Xe)
        negative_samples_generator = negatives.BernoulliNegativeSamplesGenerator(
            index_generator=random_index_generator, candidate_indices=candidate_negative_indices,
            random_state=random_state, ps_count=ps_count, po_count=po_count)
    else:
        raise ValueError("Unknown negative samples generator: %s" % negatives_name)

    nb_sample_sets = negative_samples_generator.nb_sample_sets + 1

    def loss(y_true, y_pred):
        loss_kwargs = dict(
            y_true=y_true, y_pred=y_pred,
            nb_sample_sets=nb_sample_sets,
            margin=margin)
        ranking_loss = getattr(ranking_objectives, loss_name)
        return ranking_loss(**loss_kwargs)

    optimizer = optimizers.make_optimizer(optimizer_name,
                                          lr=optimizer_lr, momentum=optimizer_momentum,
                                          decay=optimizer_decay, nesterov=optimizer_nesterov,
                                          epsilon=optimizer_epsilon, rho=optimizer_rho,
                                          beta_1=optimizer_beta_1, beta_2=optimizer_beta_2)

    model.compile(loss=loss, optimizer=optimizer)

    for epoch_no in range(1, nb_epochs + 1):
        logging.info('Epoch no. %d of %d (samples: %d)' % (epoch_no, nb_epochs, nb_samples))

        # Shuffling training (positive) triples..
        order = random_state.permutation(nb_samples)
        Xr_shuffled, Xe_shuffled = Xr[order, :], Xe[order, :]

        negative_samples = negative_samples_generator(Xr_shuffled, Xe_shuffled)
        positive_negative_samples = [(Xr_shuffled, Xe_shuffled)] + negative_samples

        batches, losses = make_batches(nb_samples, batch_size), []

        # Iterate over batches of (positive) training examples
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            current_batch_size = batch_end - batch_start
            logging.debug('Batch no. %d of %d (%d:%d), size %d'
                          % (batch_index, len(batches), batch_start, batch_end, current_batch_size))

            train_Xr_batch = np.empty((current_batch_size * nb_sample_sets, Xr_shuffled.shape[1]))
            train_Xe_batch = np.empty((current_batch_size * nb_sample_sets, Xe_shuffled.shape[1]))

            for i, samples_set in enumerate(positive_negative_samples):
                (_Xr, _Xe) = samples_set
                train_Xr_batch[i::nb_sample_sets, :] = _Xr[batch_start:batch_end, :]
                train_Xe_batch[i::nb_sample_sets, :] = _Xe[batch_start:batch_end, :]

            y_batch = np.zeros(train_Xr_batch.shape[0])

            hist = model.fit([train_Xr_batch, train_Xe_batch], y_batch,
                             nb_epoch=1, batch_size=train_Xr_batch.shape[0],
                             shuffle=False, verbose=0)

            losses += [hist.history['loss'][0] / float(train_Xr_batch.shape[0])]

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
    metrics.ranking_summary(res, tag=tag, n=1)
    metrics.ranking_summary(res, tag=tag, n=3)
    metrics.ranking_summary(res, tag=tag, n=5)
    metrics.ranking_summary(res, tag=tag, n=10)
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

    if rules is not None:
        path_ranking_client = PathRankingClient(url_or_path=rules)
        pfw_triples = path_ranking_client.request(None, threshold=.0, top_k=rules_top_k)

        model_to_regularizer = dict(
            TransE=TranslationRuleRegularizer,
            ScalE=ScalingRuleRegularizer)

        rule_regularizers = []
        for rule_predicate, rule_feature, rule_weight in pfw_triples:
            if rules_threshold is None or rule_weight >= rules_threshold:
                if rules_max_length is None or len(rule_feature.hops) <= rules_max_length:
                    head = parser.predicate_index[rule_predicate]
                    tail = [(parser.predicate_index[hop.predicate], hop.is_inverse) for hop in rule_feature.hops]

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

    train_sequences = parser.facts_to_sequences(train_facts)

    model = train_model(train_sequences, nb_entities, nb_predicates, seed=seed,
                        entity_embedding_size=entity_embedding_size, predicate_embedding_size=predicate_embedding_size,

                        dropout_entity_embeddings=dropout_entity_embeddings,
                        dropout_predicate_embeddings=dropout_predicate_embeddings,

                        model_name=model_name, similarity_name=similarity_name,
                        nb_epochs=nb_epochs, batch_size=batch_size, nb_batches=nb_batches, margin=margin,
                        loss_name=loss_name, negatives_name=negatives_name,

                        optimizer_name=optimizer_name, optimizer_lr=optimizer_lr, optimizer_momentum=optimizer_momentum,
                        optimizer_decay=optimizer_decay, optimizer_nesterov=optimizer_nesterov,
                        optimizer_epsilon=optimizer_epsilon, optimizer_rho=optimizer_rho,
                        optimizer_beta_1=optimizer_beta_1, optimizer_beta_2=optimizer_beta_2,
                        rule_regularizer=rule_regularizer)

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
