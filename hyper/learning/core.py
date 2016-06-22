# -*- coding: utf-8 -*-

import math
import numpy as np

from keras.models import Sequential
from keras.layers import SimpleRNN, GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Merge, Dropout, Reshape, Dense
from keras.engine.training import make_batches

import hyper.layers.core as core

from hyper.learning import samples, negatives
from hyper import ranking_objectives, constraints

import hyper.learning.util as learning_util
import hyper.visualization.visualization as visualization

import logging


def pairwise_training(train_sequences, nb_entities, nb_predicates, seed=1,
                      entity_embedding_size=100, predicate_embedding_size=None,
                      dropout_entity_embeddings=None, dropout_predicate_embeddings=None,
                      model_name='TransE', similarity_name='L1', nb_epochs=1000, batch_size=128, nb_batches=None,
                      margin=1.0, loss_name='hinge', negatives_name='corrupt', optimizer=None, regularizer=None,
                      hidden_size=None, entity_constraint=None, predicate_constraint=None, visualize=False):

    np.random.seed(seed)
    random_state = np.random.RandomState(seed=seed)

    predicate_encoder = Sequential()
    entity_encoder = Sequential()

    if predicate_embedding_size is None:
        predicate_embedding_size = entity_embedding_size
        if model_name in ['ManifoldESphere']:
            predicate_embedding_size = entity_embedding_size + 1
        if model_name in ['DAffinE', 'ConcatE', 'DualTransE', 'DualScalE', 'ScalTransE']:
            predicate_embedding_size = entity_embedding_size * 2
        if model_name in ['ManifoldEHyperplane']:
            predicate_embedding_size = (entity_embedding_size * 2) + 1
        elif model_name in ['BilinearE', 'RESCAL']:
            predicate_embedding_size = entity_embedding_size ** 2
        elif model_name in ['DualBilinearE', 'DualRESCAL']:
            predicate_embedding_size = (entity_embedding_size ** 2) * 2
        elif model_name in ['AffinE']:
            predicate_embedding_size = (entity_embedding_size ** 2) + entity_embedding_size
        elif model_name in ['DualAffinE']:
            predicate_embedding_size = ((entity_embedding_size ** 2) + entity_embedding_size) * 2

    predicate_input_length, entity_input_length = None, None
    if model_name == 'ER-MLP':
        predicate_input_length, entity_input_length = 1, 2

    predicate_embedding_layer = Embedding(input_dim=nb_predicates + 1, output_dim=predicate_embedding_size,
                                          input_length=predicate_input_length, init='glorot_uniform',
                                          W_regularizer=regularizer, W_constraint=predicate_constraint)
    predicate_encoder.add(predicate_embedding_layer)

    if dropout_predicate_embeddings is not None and dropout_predicate_embeddings > .0:
        predicate_encoder.add(Dropout(dropout_predicate_embeddings))

    entity_constraints, norm_constraint = None, constraints.NormConstraint(m=1., axis=1)
    if entity_constraint is None:
        entity_constraints = norm_constraint
    else:
        entity_constraints = constraints.GroupConstraint(constraints=[entity_constraint, norm_constraint])

    entity_embedding_layer = Embedding(input_dim=nb_entities + 1, output_dim=entity_embedding_size,
                                       input_length=entity_input_length, init='glorot_uniform',
                                       W_constraint=entity_constraints)
    entity_encoder.add(entity_embedding_layer)

    if dropout_entity_embeddings is not None and dropout_entity_embeddings > .0:
        entity_encoder.add(Dropout(dropout_entity_embeddings))

    model = Sequential()

    setattr(core, 'similarity function', similarity_name)
    setattr(core, 'merge function', model_name)

    if model_name in ['TransE', 'DualTransE', 'ScalE', 'ScalEQ', 'DualScalE', 'DAffinE', 'DualDAffinE', 'ScalTransE',
                      'ConcatE', 'HolE', 'ManifoldESphere', 'ManifoldEHyperplane',
                      'BilinearE', 'DualBilinearE', 'RESCAL', 'DualRESCAL', 'AffinE', 'DualAffinE']:
        merge_function = core.latent_distance_binary_merge_function
        merge_layer = Merge([predicate_encoder, entity_encoder], mode=merge_function, output_shape=lambda _: (None, 1))
        model.add(merge_layer)

    elif model_name in ['ER-MLP']:
        reshape_layer = Reshape(target_shape=(1, entity_embedding_size * entity_input_length))
        entity_encoder.add(reshape_layer)

        model = Sequential()
        merge_layer = Merge([predicate_encoder, entity_encoder], mode='concat', concat_axis=-1)
        model.add(merge_layer)

        reshape_layer = Reshape(target_shape=(entity_embedding_size * (entity_input_length + 1),))
        model.add(reshape_layer)

        C_layer = Dense(output_dim=hidden_size, activation='tanh')
        model.add(C_layer)

        w_layer = Dense(output_dim=1)
        model.add(w_layer)
    elif model_name in ['rTransE', 'rScalE']:
        merge_function = core.latent_distance_nary_merge_function

        merge_layer = Merge([predicate_encoder, entity_encoder], mode=merge_function, output_shape=lambda _: (None, 1))
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

        merge_layer = Merge([predicate_encoder, entity_encoder], mode=merge_function, output_shape=lambda _: (None, 1))
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

    def loss(y_true, y_predicted):
        loss_kwargs = dict(y_true=y_true, y_pred=y_predicted, nb_sample_sets=nb_sample_sets, margin=margin)
        ranking_loss = getattr(ranking_objectives, loss_name)
        return ranking_loss(**loss_kwargs)

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

            train_Xr_batch = np.zeros((current_batch_size * nb_sample_sets, Xr_shuffled.shape[1]))
            train_Xe_batch = np.zeros((current_batch_size * nb_sample_sets, Xe_shuffled.shape[1]))

            for i, samples_set in enumerate(positive_negative_samples):
                (_Xr, _Xe) = samples_set
                train_Xr_batch[i::nb_sample_sets, :] = _Xr[batch_start:batch_end, :]
                train_Xe_batch[i::nb_sample_sets, :] = _Xe[batch_start:batch_end, :]

            y_batch = np.zeros(train_Xr_batch.shape[0])

            hist = model.fit([train_Xr_batch, train_Xe_batch], y_batch, nb_epoch=1, batch_size=train_Xr_batch.shape[0],
                             shuffle=False, verbose=0)

            losses += [hist.history['loss'][0] / float(train_Xr_batch.shape[0])]

        if visualize is True:
            hinton_diagram = visualization.HintonDiagram()
            W_emb = predicate_embedding_layer.trainable_weights[0].get_value()
            print('Embedding dimensions: %s - Max value: %s, Min value: %s'
                  % (str(W_emb.shape), np.max(W_emb), np.min(W_emb)))
            print(hinton_diagram(W_emb))

        logging.info('Loss: %s +/- %s' % (round(np.mean(losses), 4), round(np.std(losses), 4)))

        if np.isnan(np.mean(losses)):
            raise ValueError('NaN propagation.')

    return model
