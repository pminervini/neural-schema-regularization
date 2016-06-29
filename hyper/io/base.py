# -*- coding: utf-8 -*-

import gzip
import bz2

import logging


def iopen(file, *args, **kwargs):
    _open = open
    if file.endswith('.gz'):
        _open = gzip.open
    elif file.endswith('.bz2'):
        _open = bz2.open
    return _open(file, *args, **kwargs)


def read_triples(path):
    triples = None
    if path is not None:
        logging.info('Acquiring %s ..' % path)
        with iopen(path, 'rt') as f:
            lines = f.readlines()
        triples = [(s.strip(), p.strip(), o.strip()) for [s, p, o] in [l.split() for l in lines]]
    return triples
