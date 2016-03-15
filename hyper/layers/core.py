# -*- coding: utf-8 -*-

from keras import backend as K


def latent_distance_merge_function(args):
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
    relation_embedding, entity_embeddings = args[0], args[1]
    #pred = relation_embedding[:, 0, :]
    #subj, obj = entity_embeddings[:, 0, :], entity_embeddings[:, 1, :]
    return K.concatenate([relation_embedding, entity_embeddings], axis=1)
