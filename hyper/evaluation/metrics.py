# -*- coding: utf-8 -*-

import numpy as np


def ranking_score(scoring_function, triples, max_subj_idx, max_obj_idx):
    err_subj, err_obj = [], []

    for subj_idx, pred_idx, obj_idx in triples:
        Xr = np.empty((max_subj_idx, 1))
        Xr[:, 0] = pred_idx

        Xe = np.empty((max_subj_idx, 2))
        Xe[:, 0] = np.arange(1, max_subj_idx + 1)
        Xe[:, 1] = obj_idx

        scores_left = scoring_function([Xr, Xe])

        err_subj += [np.argsort(np.argsort(scores_left.flatten())[::-1])[subj_idx - 1] + 1]

        Xr = np.empty((max_obj_idx, 1))
        Xr[:, 0] = pred_idx

        Xe = np.empty((max_obj_idx, 2))
        Xe[:, 0] = subj_idx
        Xe[:, 1] = np.arange(1, max_obj_idx + 1)

        scores_right = scoring_function([Xr, Xe])

        err_obj += [np.argsort(np.argsort(scores_right.flatten())[::-1])[obj_idx - 1] + 1]

    return err_subj, err_obj


def filtered_ranking_score(scoring_function, triples, max_subj_idx, max_obj_idx, true_triples):
    err_subj, err_obj = [], []

    for subj_idx, pred_idx, obj_idx in triples:
        _is = np.argwhere(true_triples[:, 0] == subj_idx).reshape(-1,)
        _ip = np.argwhere(true_triples[:, 1] == pred_idx).reshape(-1,)
        _io = np.argwhere(true_triples[:, 2] == obj_idx).reshape(-1,)

        Xr = np.empty((max_subj_idx, 1))
        Xr[:, 0] = pred_idx

        Xe = np.empty((max_subj_idx, 2))
        Xe[:, 0] = np.arange(1, max_subj_idx + 1)
        Xe[:, 1] = obj_idx

        scores_left = scoring_function([Xr, Xe])

        inter_subj = [i for i in _io if i in _ip]
        rmv_idx_subj = [true_triples[i, 0] - 1 for i in inter_subj if true_triples[i, 0] != subj_idx]
        scores_left[rmv_idx_subj] = - np.inf

        err_subj += [np.argsort(np.argsort(scores_left.flatten())[::-1])[subj_idx - 1] + 1]

        Xr = np.empty((max_obj_idx, 1))
        Xr[:, 0] = pred_idx

        Xe = np.empty((max_obj_idx, 2))
        Xe[:, 0] = subj_idx
        Xe[:, 1] = np.arange(1, max_obj_idx + 1)

        scores_right = scoring_function([Xr, Xe])

        inter_obj = [i for i in _is if i in _ip]
        rmv_idx_obj = [true_triples[i, 2] - 1 for i in inter_obj if true_triples[i, 2] != obj_idx]
        scores_right[rmv_idx_obj] = - np.inf

        err_obj += [np.argsort(np.argsort(scores_right.flatten())[::-1])[obj_idx - 1] + 1]

    return err_subj, err_obj
