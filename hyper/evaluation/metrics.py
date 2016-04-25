# -*- coding: utf-8 -*-

import numpy as np
import logging


def ranking_score(scoring_function, triples, max_subj_idx, max_obj_idx):
    err_subj, err_obj = [], []

    for subj_idx, pred_idx, obj_idx in triples:
        Xr = np.empty((max_subj_idx, 1))
        Xr[:, 0] = pred_idx

        Xe = np.empty((max_subj_idx, 2))
        Xe[:, 0] = np.arange(1, max_subj_idx + 1)
        Xe[:, 1] = obj_idx

        scores_left = scoring_function([Xr, Xe])

        err_subj += [np.argsort(np.argsort(- scores_left))[subj_idx - 1] + 1]

        Xr = np.empty((max_obj_idx, 1))
        Xr[:, 0] = pred_idx

        Xe = np.empty((max_obj_idx, 2))
        Xe[:, 0] = subj_idx
        Xe[:, 1] = np.arange(1, max_obj_idx + 1)

        scores_right = scoring_function([Xr, Xe])

        err_obj += [np.argsort(np.argsort(- scores_right))[obj_idx - 1] + 1]

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


def ranking_summary(res, idxo=None, n=10, tag='raw'):
    dres = {}
    dres.update({'microlmean': np.mean(res[0])})
    dres.update({'microlmedian': np.median(res[0])})
    dres.update({'microlhits@n': np.mean(np.asarray(res[0]) <= n) * 100})
    dres.update({'micrormean': np.mean(res[1])})
    dres.update({'micrormedian': np.median(res[1])})
    dres.update({'microrhits@n': np.mean(np.asarray(res[1]) <= n) * 100})

    resg = res[0] + res[1]

    dres.update({'microgmean': np.mean(resg)})
    dres.update({'microgmedian': np.median(resg)})
    dres.update({'microghits@n': np.mean(np.asarray(resg) <= n) * 100})

    dres.update({'microlmrr': np.mean(1. / np.array(res[0]))})
    dres.update({'micrormrr': np.mean(1. / np.array(res[1]))})
    dres.update({'microgmrr': np.mean(1. / resg)})

    logging.info('### MICRO (%s):' % (tag))
    logging.info('\t-- left   >> mean: %s, median: %s, mrr: %s, hits@%s: %s%%' %
                 (round(dres['microlmean'], 5), round(dres['microlmedian'], 5),
                  round(dres['microlmrr'], 3), n, round(dres['microlhits@n'], 3)))
    logging.info('\t-- right  >> mean: %s, median: %s, mrr: %s, hits@%s: %s%%' %
                 (round(dres['micrormean'], 5), round(dres['micrormedian'], 5),
                  round(dres['micrormrr'], 3), n, round(dres['microrhits@n'], 3)))
    logging.info('\t-- global >> mean: %s, median: %s, mrr: %s, hits@%s: %s%%' %
                 (round(dres['microgmean'], 5), round(dres['microgmedian'], 5),
                  round(dres['microgmrr'], 3), n, round(dres['microghits@n'], 3)))

    if idxo is not None:
        listrel = set(idxo)
        dictrelres = {}
        dictrellmean = {}
        dictrelrmean = {}
        dictrelgmean = {}
        dictrellmedian = {}
        dictrelrmedian = {}
        dictrelgmedian = {}
        dictrellrn = {}
        dictrelrrn = {}
        dictrelgrn = {}

        for i in listrel:
            dictrelres.update({i: [[], []]})

        for i, j in enumerate(res[0]):
            dictrelres[idxo[i]][0] += [j]

        for i, j in enumerate(res[1]):
            dictrelres[idxo[i]][1] += [j]

        for i in listrel:
            dictrellmean[i] = np.mean(dictrelres[i][0])
            dictrelrmean[i] = np.mean(dictrelres[i][1])
            dictrelgmean[i] = np.mean(dictrelres[i][0] + dictrelres[i][1])
            dictrellmedian[i] = np.median(dictrelres[i][0])
            dictrelrmedian[i] = np.median(dictrelres[i][1])
            dictrelgmedian[i] = np.median(dictrelres[i][0] + dictrelres[i][1])
            dictrellrn[i] = np.mean(np.asarray(dictrelres[i][0]) <= n) * 100
            dictrelrrn[i] = np.mean(np.asarray(dictrelres[i][1]) <= n) * 100
            dictrelgrn[i] = np.mean(np.asarray(dictrelres[i][0] + dictrelres[i][1]) <= n) * 100

        dres.update({'dictrelres': dictrelres})
        dres.update({'dictrellmean': dictrellmean})
        dres.update({'dictrelrmean': dictrelrmean})
        dres.update({'dictrelgmean': dictrelgmean})
        dres.update({'dictrellmedian': dictrellmedian})
        dres.update({'dictrelrmedian': dictrelrmedian})
        dres.update({'dictrelgmedian': dictrelgmedian})

        dres.update({'dictrellrn': dictrellrn})
        dres.update({'dictrelrrn': dictrelrrn})
        dres.update({'dictrelgrn': dictrelgrn})

        dres.update({'macrolmean': np.mean(dictrellmean.values())})
        dres.update({'macrolmedian': np.mean(dictrellmedian.values())})
        dres.update({'macrolhits@n': np.mean(dictrellrn.values())})
        dres.update({'macrormean': np.mean(dictrelrmean.values())})
        dres.update({'macrormedian': np.mean(dictrelrmedian.values())})
        dres.update({'macrorhits@n': np.mean(dictrelrrn.values())})
        dres.update({'macrogmean': np.mean(dictrelgmean.values())})
        dres.update({'macrogmedian': np.mean(dictrelgmedian.values())})
        dres.update({'macroghits@n': np.mean(dictrelgrn.values())})

        logging.info('### MACRO (%s):' % (tag))
        logging.info('\t-- left   >> mean: %s, median: %s, hits@%s: %s%%' %
                     (round(dres['macrolmean'], 5), round(dres['macrolmedian'], 5),
                      n, round(dres['macrolhits@n'], 3)))
        logging.info('\t-- right  >> mean: %s, median: %s, hits@%s: %s%%' %
                     (round(dres['macrormean'], 5), round(dres['macrormedian'], 5),
                      n, round(dres['macrorhits@n'], 3)))
        logging.info('\t-- global >> mean: %s, median: %s, hits@%s: %s%%' %
                     (round(dres['macrogmean'], 5), round(dres['macrogmedian'], 5),
                      n, round(dres['macroghits@n'], 3)))

    return dres
