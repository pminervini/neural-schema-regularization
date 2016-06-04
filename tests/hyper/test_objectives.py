# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K
from hyper import objectives

import math
import unittest


def loss(a, b, f):
    return f(a, b)


class TestObjectives(unittest.TestCase):
    nb_samples = 100
    rs = None

    def setUp(self):
        self.rs = np.random.RandomState(0)

    def test_objectives(self):
        for loss_f in [objectives.MSE, objectives.RMSE, objectives.MAE, objectives.MAPE, objectives.MSLE,
                       objectives.SHL, objectives.HL, objectives.BC, objectives.PL]:
            for _ in range(16):
                p = np.random.random(self.nb_samples).astype(K.floatx())
                t = np.random.random_integers(0, 1, self.nb_samples).astype(K.floatx())

                P, T = K.placeholder(ndim=1), K.placeholder(ndim=1)
                function = K.function([T, P], [loss(T, P, f=loss_f)])

                d = function([t, p])[0]
                s = function([t, t])[0]

                self.assertTrue(d > s)

    def test_mse(self):
        p_values = self.rs.rand(1024)
        t_values = self.rs.randint(0, 2, 1024)

        loss_f = objectives.MSE

        P, T = K.placeholder(ndim=1), K.placeholder(ndim=1)
        function = K.function([T, P], [loss(T, P, f=loss_f)])

        for p, t in zip(p_values, t_values):
            self.assertTrue(abs(function([[t], [p]])[0] - ((t - p) ** 2)) < 1e-6)

    def test_rmse(self):
        p_values = self.rs.rand(1024)
        t_values = self.rs.randint(0, 2, 1024)

        loss_f = objectives.RMSE

        P, T = K.placeholder(ndim=1), K.placeholder(ndim=1)
        function = K.function([T, P], [loss(T, P, f=loss_f)])

        for p, t in zip(p_values, t_values):
            self.assertTrue(abs(function([[t], [p]])[0] - math.sqrt((t - p) ** 2)) < 1e-6)

    def test_mae(self):
        p_values = self.rs.rand(1024)
        t_values = self.rs.randint(0, 2, 1024)

        loss_f = objectives.RMSE

        P, T = K.placeholder(ndim=1), K.placeholder(ndim=1)
        function = K.function([T, P], [loss(T, P, f=loss_f)])

        for p, t in zip(p_values, t_values):
            self.assertTrue(abs(function([[t], [p]])[0] - abs(t - p)) < 1e-6)

    def test_bc(self):
        p_values = self.rs.rand(1024)
        t_values = self.rs.randint(0, 2, 1024)

        loss_f = objectives.BC

        P, T = K.placeholder(ndim=1), K.placeholder(ndim=1)
        function = K.function([T, P], [loss(T, P, f=loss_f)])

        for p, t in zip(p_values, t_values):
            cp = np.clip(p, 10e-8, 1.0 - 10e-8)
            bc = - (t * math.log(cp) + (1. - t) * math.log(1. - cp))
            self.assertTrue(abs(function([[t], [p]])[0] - bc) < 1e-2)

if __name__ == '__main__':
    unittest.main()
