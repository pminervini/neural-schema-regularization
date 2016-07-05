#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_command(c):
    command = "PYTHONPATH=. ./bin/hyper-cli.py" \
              " --train data/wn18/wordnet-mlj12-train.txt" \
              " --valid data/wn18/wordnet-mlj12-valid.txt" \
              " --test data/wn18/wordnet-mlj12-test.txt" \
              " --epochs %s" \
              " --optimizer %s" \
              " --lr %s" \
              " --batches %s" \
              " --model %s" \
              " --similarity %s" \
              " --margin %s" \
              " --entity-embedding-size %s" \
              " --entity-rank %s" \
              % (c['epochs'], c['optimizer'], c['lr'], c['batches'], c['model'], c['similarity'], c['margin'],
                 c['emb_size'], c['emb_rank'])
    return command


def to_logfile(c, dir):
    outfile = "%s/exp_wn18_lowrank_v1.%s.log" % (dir, summary(c))
    return outfile


hyperparameters_space = dict(
    epochs=[500],
    optimizer=['adagrad'],
    lr=[.1],
    batches=[10],
    model=['TransE', 'ScalE'],
    similarity=['l1', 'l2', 'dot'],
    margin=[1],
    emb_size=[20, 50, 100, 200, 300, 400],
    emb_rank=[20, 50, 100, 200, 300, 400],
)

configurations = cartesian_product(hyperparameters_space)

dir = 'logs/exp_wn18_lowrank_v1/'

for c in configurations:
    if c['emb_rank'] <= c['emb_size']:
        logfile = to_logfile(c, dir)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r') as f:
                content = f.read()
                completed = ('### MICRO (test filtered)' in content) or ('### COMPLETED' in content)

        if not completed:
            line = '%s >> %s 2>&1' % (to_command(c), logfile)
            print(line)
