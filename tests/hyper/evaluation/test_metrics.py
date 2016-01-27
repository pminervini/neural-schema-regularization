# -*- coding: utf-8 -*-

import numpy as np
from hyper.evaluation import metrics

import unittest


def scoring_function(args):
    Xr, Xe = args[0], args[1]
    dict = {
        (1, 1, 1): 1.0,
        (1, 1, 2): 2.0,
        (1, 1, 3): 3.0,
        (2, 1, 1): 0.7,
        (2, 1, 2): 0.9,
        (2, 1, 3): 1.1,
        (3, 1, 1): 1.3,
        (3, 1, 2): 1.5,
        (3, 1, 3): 1.7,

        (4, 1, 1): 0.0,
        (4, 1, 2): 0.0,
        (4, 1, 3): 0.0,
        (4, 1, 4): 0.0,
        (1, 1, 4): 0.0,
        (2, 1, 4): 0.0,
        (3, 1, 4): 0.0,
    }
    values = []
    for i in range(Xr.shape[0]):
        subj, pred, obj = Xe[i, 0], Xr[i, 0], Xe[i, 1]
        values += [dict[(subj, pred, obj)]]
    return np.array(values)


class TestMetrics(unittest.TestCase):

    def setUp(self):
        pass

    def test_ranking_score(self):
        true_triples = np.empty((0, 3))

        err_subj, err_obj = metrics.ranking_score(scoring_function, [(1, 1, 1)], 4, 4)
        self.assertTrue(err_subj[0] == 2 and err_obj[0] == 3)
        err_subj, err_obj = metrics.filtered_ranking_score(scoring_function, [(1, 1, 1)], 4, 4, true_triples)
        self.assertTrue(err_subj[0] == 2 and err_obj[0] == 3)

        err_subj, err_obj = metrics.ranking_score(scoring_function, [(1, 1, 2)], 4, 4)
        self.assertTrue(err_subj[0] == 1 and err_obj[0] == 2)
        err_subj, err_obj = metrics.filtered_ranking_score(scoring_function, [(1, 1, 2)], 4, 4, true_triples)
        self.assertTrue(err_subj[0] == 1 and err_obj[0] == 2)

        err_subj, err_obj = metrics.ranking_score(scoring_function, [(2, 1, 1)], 4, 4)
        self.assertTrue(err_subj[0] == 3 and err_obj[0] == 3)
        err_subj, err_obj = metrics.filtered_ranking_score(scoring_function, [(2, 1, 1)], 4, 4, true_triples)
        self.assertTrue(err_subj[0] == 3 and err_obj[0] == 3)

    def test_filtered_ranking_score(self):
        true_triples = np.array([[1, 1, 3]])

        err_subj, err_obj = metrics.filtered_ranking_score(scoring_function, [(1, 1, 1)], 4, 4, true_triples)
        self.assertTrue(err_subj[0] == 2 and err_obj[0] == 2)

        err_subj, err_obj = metrics.filtered_ranking_score(scoring_function, [(1, 1, 2)], 4, 4, true_triples)
        self.assertTrue(err_subj[0] == 1 and err_obj[0] == 1)

        err_subj, err_obj = metrics.filtered_ranking_score(scoring_function, [(2, 1, 1)], 4, 4, true_triples)
        self.assertTrue(err_subj[0] == 3 and err_obj[0] == 3)

        true_triples = np.array([[1, 1, 3], [3, 1, 1]])

        err_subj, err_obj = metrics.filtered_ranking_score(scoring_function, [(1, 1, 1)], 4, 4, true_triples)
        self.assertTrue(err_subj[0] == 1 and err_obj[0] == 2)

        err_subj, err_obj = metrics.filtered_ranking_score(scoring_function, [(1, 1, 2)], 4, 4, true_triples)
        self.assertTrue(err_subj[0] == 1 and err_obj[0] == 1)

        err_subj, err_obj = metrics.filtered_ranking_score(scoring_function, [(2, 1, 1)], 4, 4, true_triples)
        self.assertTrue(err_subj[0] == 2 and err_obj[0] == 3)

        true_triples = np.array([[1, 1, 3], [3, 1, 1], [1, 1, 1]])

        err_subj, err_obj = metrics.filtered_ranking_score(scoring_function, [(1, 1, 1)], 4, 4, true_triples)
        self.assertTrue(err_subj[0] == 1 and err_obj[0] == 2)


if __name__ == '__main__':
    unittest.main()
