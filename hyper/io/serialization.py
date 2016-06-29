# -*- coding: utf-8 -*-


def serialize(prefix, model, parser):

    # Saving the weights of the model
    model_path = '%s_weights.h5' % prefix
    model.save_weights(model_path, overwrite=True)

    # Saving the parser
    import pickle
    parser_path = '%s_parser.p' % prefix
    with open(parser_path, 'wb') as f:
        pickle.dump(parser, f)

    return
