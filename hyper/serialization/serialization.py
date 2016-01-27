# -*- coding: utf-8 -*-

import os
import pickle


def serialize(model, folder_path, entity2idx, predicate2idx):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    json_model = model.to_json()

    architecture_path = os.path_join(folder_path, 'architecture.json')
    weights_path = os.path_join(folder_path, 'weights.h5')
    dictionaries_path = os.path.join(folder_path, 'dictionaries.pkl')

    with open(architecture_path, 'w') as f:
        f.write(json_model)

    model.save_weights(weights_path)

    with open(dictionaries_path, 'w') as f:
        pickle.dump({'entity2idx': entity2idx, 'predicate2idx': predicate2idx}, f)

    return
