# -*- coding: utf-8 -*-

from keras import backend as K


def latent_distance_merge_function(args):
    """
    Takes a list args=[Xr, Xe], where Xr is a batch_size x 1 x embedding_size
    Tensor, and Xe is a batch_size x 2 x embedding_size tensor: first it obtains
    a batch_size x embedding_size Tensor A from Xe, and then computes the
    similarities between each column of A and Xr[:, 0, :].

    :param args: List of tensors.
    :return: batch_size x 1 Tensor of similarity values.
    """
    import sys
    import hyper.similarities as similarities
    import hyper.layers.binary.merge_functions as merge_functions

    f_core = sys.modules['hyper.layers.core']

    similarity_function_name = getattr(f_core, 'similarity function')
    merge_function_name = getattr(f_core, 'merge function')

    similarity_function = similarities.get_function(similarity_function_name)
    merge_function = merge_functions.get_function(merge_function_name)

    return merge_function(args, similarity=similarity_function)


def concatenate_embeddings_merge_function(args):
    """
    Takes a list args=[Xr, Xe], where Xr is a batch_size x 1 x embedding_size
    Tensor, and Xe is a batch_size x 2 x embedding_size tensor, a concatenates
    them on the first dimension, obtaining a batch_size x 3 x embedding_size
    tensor.

    :param args: List of tensors.
    :return: batch_size x 3 x embedding_size Tensor.
    """
    relation_embedding, entity_embeddings = args[0], args[1]
    return K.concatenate([relation_embedding, entity_embeddings], axis=1)


def similarity_merge_function(args):
    """
    Takes a list args=[Xr, Xe], where Xr is a batch_size x 1 x embedding_size
    Tensor, and Xe is a batch_size x embedding_size tensor, and computes
    the similarities between Xr[:, 0, :] and Xe.

    Xe is meant to embed the original sequence of entities, e.g. by means of
    a recurrent neural network architecture.

    :param args: List of tensors.
    :return: batch_size x 1 Tensor of similarity values.
    """
    import sys
    import hyper.similarities as similarities

    f_core = sys.modules['hyper.layers.core']

    similarity_function_name = getattr(f_core, 'similarity function')
    similarity_function = similarities.get_function(similarity_function_name)

    relation_embedding, entity_embedding = args[0], args[1]
    sim = similarity_function(relation_embedding[:, 0, :], entity_embedding, axis=-1)

    return K.reshape(sim, (-1, 1))
