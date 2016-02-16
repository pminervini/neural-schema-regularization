#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_command(c):
    command = "PYTHONPATH=. ./bin/hyper-cli.py" \
              " --train data/fb15k/freebase_mtr100_mte100-train.txt" \
              " --valid data/fb15k/freebase_mtr100_mte100-valid.txt" \
              " --test data/fb15k/freebase_mtr100_mte100-test.txt" \
              " --epochs %s" \
              " --optimizer %s" \
              " --lr %s" \
              " --batches %s" \
              " --model %s" \
              " --similarity %s" \
              " --entity-embedding-size %s --predicate-embedding-size %s"\
              % (c['epochs'], c['optimizer'], c['lr'], c['batches'],
                 c['model'], c['similarity'], c['embedding_size'], c['embedding_size'])
    return command + " >> logs/exp_fb15k_v1." + summary(c) + ".log 2>&1"


hyperparameters_space = dict(
    epochs=[500],
    optimizer=['adagrad'],
    lr=[.001, .01, .1, 1.],
    batches=[10],
    model=['TransE', 'ScalE'],
    similarity=['l1', 'l2', 'dot'],
    embedding_size=[20, 50, 100, 200, 300, 400])

configurations = cartesian_product(hyperparameters_space)

for configuration in configurations:
    print(to_command(configuration))




