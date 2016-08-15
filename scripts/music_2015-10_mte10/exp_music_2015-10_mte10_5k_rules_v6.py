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
              " --train data/music_2015-10_mte10_5k/music_2015-10_mte10-train.tsv.gz" \
              " --valid data/music_2015-10_mte10_5k/music_2015-10_mte10-valid.tsv.gz" \
              " --test data/music_2015-10_mte10_5k/music_2015-10_mte10-test.tsv.gz" \
              " --epochs %s" \
              " --optimizer %s" \
              " --lr %s" \
              " --batches %s" \
              " --model %s" \
              " --similarity %s" \
              " --margin %s" \
              " --entity-embedding-size %s" \
              " --rules ./data/music_2015-10_mte10_5k/rules/music_2015-10_mte10-rules.json.gz" \
              " --rules-threshold %s" \
              " --sample-facts %s" \
              " --rules-lambda %s" \
              % (c['epochs'], c['optimizer'], c['lr'], c['batches'],
                 c['model'], c['similarity'], c['margin'],
                 c['embedding_size'],
                 c['rules_threshold'], c['sample_facts'], c['rules_lambda'])
    return command


def to_logfile(c, dir):
    outfile = "%s/exp_music_2015-10_mte10_5k_rules_v6.%s.log" % (dir, summary(c))
    return outfile

hyperparameters_space = dict(
    epochs=[100],
    optimizer=['adagrad'],
    lr=[.1],
    batches=[10],
    model=['ComplEx'],
    similarity=['dot'],
    margin=[1],
    embedding_size=[10, 20, 50, 100, 150, 200, 300],

    rules_threshold=[.9],
    sample_facts=[1],
    rules_lambda=[.0, .00001, .0001, .001, .01, .1, 1, 10, 100, 1000, 10000, 100000])

configurations = cartesian_product(hyperparameters_space)

dir = 'logs/exp_music_2015-10_mte10_5k_rules_v6/'

for c in configurations:
    logfile = to_logfile(c, dir)

    completed = False
    if os.path.isfile(logfile):
        with open(logfile, 'r') as f:
            content = f.read()
            completed = '### MICRO (test filtered)' in content

    if not completed:
        line = '%s >> %s 2>&1' % (to_command(c), logfile)
        print(line)
