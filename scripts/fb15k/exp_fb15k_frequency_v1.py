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
    freq_one, freq_two = c['freq_emb_len_one'], c['freq_emb_len_two']
    command = "PYTHONPATH=. ./bin/hyper-cli.py --filtered --frequency-mask-type 1" \
              " --train data/fb15k/freebase_mtr100_mte100-train.txt" \
              " --valid data/fb15k/freebase_mtr100_mte100-valid.txt" \
              " --test data/fb15k/freebase_mtr100_mte100-test.txt" \
              " --epochs %s" \
              " --optimizer %s" \
              " --lr %s" \
              " --batches %s" \
              " --model %s" \
              " --similarity %s" \
              " --margin %s" \
              " --entity-embedding-size %s " \
              " --frequency-cutoffs %s --frequency-embedding-lengths %s %s" \
              % (c['epochs'], c['optimizer'], c['lr'], c['batches'],
                 c['model'], c['similarity'], c['margin'], max([freq_one, freq_two]),
                 c['freq_cutoffs'], freq_one, freq_two)
    return command


def to_logfile(c, dir):
    outfile = "%s/exp_fb15k_frequency_v1.%s.log" % (dir, summary(c))
    return outfile


hyperparameters_space = dict(
    epochs=[500],
    optimizer=['adagrad'],
    lr=[.1],
    batches=[10],
    model=['TransE', 'ScalE'],
    similarity=['l1', 'dot'], # , 'l2'],
    margin=[1],
    freq_emb_len_one=[20, 50, 100, 200],
    freq_emb_len_two=[20, 50, 100, 200],
    freq_cutoffs=[0, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
)

configurations = cartesian_product(hyperparameters_space)

dir = 'logs/exp_fb15k_frequency_v1/'

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
