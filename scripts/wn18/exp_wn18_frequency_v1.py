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
              " --entity-embedding-size %s --predicate-embedding-size %s" \
              " --frequency-cutoffs %s --frequency-embedding-lengths %s %s" \
              % (c['epochs'], c['optimizer'], c['lr'], c['batches'],
                 c['model'], c['similarity'], c['margin'], c['embedding_size'], c['embedding_size'],
                 c['frequency_cutoffs'], c['freq_emb_len_one'], c['freq_emb_len_two'])
    return command


def to_logfile(c, dir):
    outfile = "%s/exp_wn18_frequency_v1.%s.log" % (dir, summary(c))
    return outfile


hyperparameters_space = dict(
    epochs=[500],
    optimizer=['adagrad'],
    lr=[.1],
    batches=[10],
    model=['TransE', 'ScalE'],
    similarity=['l1', 'l2', 'dot'],
    margin=[1],
    embedding_size=[20, 50, 100, 200],
    freq_emb_len_one=[20, 50, 100, 200],
    freq_emb_len_two=[20, 50, 100, 200],
    freq_cutoffs=[50, 100, 200, 300, 400, 500]
)

configurations = cartesian_product(hyperparameters_space)

dir = 'logs/exp_wn18_frequency_v1/'

for c in configurations:
    logfile = to_logfile(c, dir)

    completed = False
    if os.path.isfile(logfile):
        with open(logfile, 'r') as f:
            content = f.read()
            completed = ('### MICRO (test filtered)' in content) or ('### COMPLETED' in content)

    if not completed:
        line = '%s >> %s 2>&1' % (to_command(c), logfile)
        print(line)
