# -*- coding: utf-8 -*-


class Hop(object):
    def __init__(self, predicate, is_inverse=False):
        self.predicate = predicate
        self.is_inverse = is_inverse

    def __str__(self):
        return self.predicate + ('^-1' if self.is_inverse is True else '')


class Feature(object):
    def __init__(self, hops):
        self.hops = hops

    def __str__(self):
        return ' . '.join([str(hop) for hop in self.hops])
