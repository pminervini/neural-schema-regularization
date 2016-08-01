# -*- coding: utf-8 -*-

import pickle


def serialize(prefix, model=None, parser=None, argv=None):

    if model is not None:
        # Saving the weights of the model
        model_path = '%s_weights.h5' % prefix
        model.save_weights(model_path, overwrite=True)

    if parser is not None:
        # Saving the fact parser
        parser_path = '%s_parser.p' % prefix
        with open(parser_path, 'wb') as f:
            pickle.dump(parser, f)

    if argv is not None:
        # Saving the command line
        readme_path = "%s_README.md" % prefix

        content = """
        # Model Details

        This model was created by using the following command line:

        ```
        ./bin/hyper-cli.py %s
        ```
        """ % ' '.join(argv)

        with open(readme_path, 'w') as f:
            f.write(content)

    return
